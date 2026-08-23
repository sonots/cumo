// Compacts the indices of the one (or, inverted, zero) bits, in ascending
// order: each block counts the hits of a contiguous range of chunks, one scan
// block turns the counts into output positions, and each block scatters its
// range from its position. *running carries the output cursor from one ndloop
// iteration to the next entirely on the device, so no launch reads anything
// back to the host.

// The scan kernel is one block, so it bounds the scratch the caller allocates
// and the sequential work it does itself.
#define CUMO_BIT_WHERE_MAX_BLOCKS 1024

__global__ void cumo_bit_where_partial_kernel(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, uint64_t nw, int contiguous, int invert, uint64_t nchunks, uint64_t cpb, uint64_t *block_sums)
{
    uint64_t start = blockIdx.x * cpb;
    uint64_t end = (nchunks - start < cpb) ? nchunks : start + cpb;
    uint64_t cnt = 0;
    uint64_t c0, total;

    if (start > nchunks) { start = end = nchunks; }
    for (c0 = start; c0 < end; c0 += blockDim.x) {
        uint64_t c = c0 + threadIdx.x;
        if (c < end) {
            cnt += (uint64_t)__popc(cumo_bit_chunk(a, p, s, idx, n, nw, c, contiguous, invert));
        }
    }
    cumo_bit_block_exscan(cnt, &total);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = total;
    }
}

__global__ void cumo_bit_where_scan_kernel(uint64_t *block_sums, uint64_t nblocks, uint64_t *running)
{
    uint64_t acc = *running;
    uint64_t t0;

    for (t0 = 0; t0 < nblocks; t0 += blockDim.x) {
        uint64_t t = t0 + threadIdx.x;
        uint64_t v = (t < nblocks) ? block_sums[t] : 0;
        uint64_t total;
        uint64_t pre = cumo_bit_block_exscan(v, &total);
        if (t < nblocks) {
            block_sums[t] = acc + pre;
        }
        acc += total;
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *running = acc;
    }
}

// The scatter takes an element per lane rather than a chunk, so that the lanes
// that do write land next to each other. A thread that owned a whole chunk wrote
// a run of its own, which put the lanes of a warp a run apart.
__global__ void cumo_bit_where_scatter_kernel(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, uint64_t nw, int contiguous, int invert, uint64_t nchunks, uint64_t cpb, char *out, size_t elmsz, uint64_t count, uint64_t *block_sums)
{
    uint64_t startc = blockIdx.x * cpb;
    uint64_t endc = (nchunks - startc < cpb) ? nchunks : startc + cpb;
    uint64_t off = block_sums[blockIdx.x];
    uint64_t start, end, i0;

    if (startc > nchunks) { startc = endc = nchunks; }
    start = startc * CUMO_NB;
    end = (endc * CUMO_NB < n) ? endc * CUMO_NB : n;
    for (i0 = start; i0 < end; i0 += blockDim.x) {
        uint64_t i = i0 + threadIdx.x;
        CUMO_BIT_DIGIT x = 0;
        uint64_t total;
        uint64_t pre;

        if (i < end) {
            CUMO_LOAD_BIT(a, cumo_bit_pos(p, s, idx, i), x);
            if (invert) x = !x;
        }
        pre = cumo_bit_block_exscan(x, &total);
        if (x) {
            cumo_bit_store_index(out, elmsz, off + pre, count + i);
        }
        off += total;
        __syncthreads();
    }
}

char *cumo_bit_where_scratch_new(void)
{
    char *scratch = cumo_cuda_runtime_malloc((2 + CUMO_BIT_WHERE_MAX_BLOCKS) * sizeof(uint64_t));
    cudaMemsetAsync(scratch, 0, 2 * sizeof(uint64_t), 0);
    cumo_cuda_runtime_check_kernel_launch();
    return scratch;
}

void cumo_bit_where_kernel_launch(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, int invert, char *out, size_t elmsz, uint64_t count, char *scratch)
{
    uint64_t nchunks = (n + CUMO_NB - 1) / CUMO_NB;
    int contiguous = (idx == NULL && s == 1);
    uint64_t nw = contiguous ? (p + n + CUMO_NB - 1) / CUMO_NB : 0;
    uint64_t nblocks = (nchunks + CUMO_BIT_CHUNK_BLOCK - 1) / CUMO_BIT_CHUNK_BLOCK;
    uint64_t *running = (uint64_t*)scratch + (invert ? 1 : 0);
    uint64_t *block_sums = (uint64_t*)scratch + 2;
    uint64_t cpb;

    if (nblocks > CUMO_BIT_WHERE_MAX_BLOCKS) nblocks = CUMO_BIT_WHERE_MAX_BLOCKS;
    cpb = (nchunks + nblocks - 1) / nblocks;
    cumo_bit_where_partial_kernel<<<nblocks, CUMO_BIT_CHUNK_BLOCK>>>(a,p,s,idx,n,nw,contiguous,invert,nchunks,cpb,block_sums);
    cumo_bit_where_scan_kernel<<<1, CUMO_BIT_CHUNK_BLOCK>>>(block_sums,nblocks,running);
    cumo_bit_where_scatter_kernel<<<nblocks, CUMO_BIT_CHUNK_BLOCK>>>(a,p,s,idx,n,nw,contiguous,invert,nchunks,cpb,out,elmsz,count,block_sums);
    cumo_cuda_runtime_check_kernel_launch();
}
