// The compaction is where's: the same partial count and block scan put every
// block's output at the right offset, and the cursor in the scratch carries
// that offset from one loop segment to the next on the device. Only the
// scatter differs -- it writes the position of the masked array's element
// rather than the flat index, so the result indexes that array's data.
__global__ void cumo_bit_mask_scatter_kernel(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, uint64_t nw, int contiguous, uint64_t nchunks, uint64_t cpb, size_t *out, size_t p2, ssize_t s2, size_t *idx2, uint64_t *block_sums)
{
    uint64_t start = blockIdx.x * cpb;
    uint64_t end = (nchunks - start < cpb) ? nchunks : start + cpb;
    uint64_t off = block_sums[blockIdx.x];
    uint64_t c0;

    if (start > nchunks) { start = end = nchunks; }
    for (c0 = start; c0 < end; c0 += blockDim.x) {
        uint64_t c = c0 + threadIdx.x;
        CUMO_BIT_DIGIT z = 0;
        uint64_t total;
        uint64_t pre;
        uint64_t o;

        if (c < end) {
            z = cumo_bit_chunk(a, p, s, idx, n, nw, c, contiguous, 0);
        }
        pre = cumo_bit_block_exscan((uint64_t)__popc(z), &total);
        o = off + pre;
        while (z) {
            uint64_t j = c * CUMO_NB + (uint64_t)(__ffs(z) - 1);
            out[o++] = idx2 ? p2 + idx2[j] : (size_t)((ssize_t)p2 + (ssize_t)j * s2);
            z &= z - 1;
        }
        off += total;
        __syncthreads();
    }
}

void cumo_bit_mask_kernel_launch(CUMO_BIT_DIGIT *a, size_t p, ssize_t s, size_t *idx, uint64_t n, size_t *out, size_t p2, ssize_t s2, size_t *idx2, char *scratch)
{
    uint64_t nchunks = (n + CUMO_NB - 1) / CUMO_NB;
    int contiguous = (idx == NULL && s == 1);
    // A grid of no blocks is an invalid launch configuration.
    if (n == 0) return;
    uint64_t nw = contiguous ? (p + n + CUMO_NB - 1) / CUMO_NB : 0;
    uint64_t nblocks = (nchunks + CUMO_BIT_CHUNK_BLOCK - 1) / CUMO_BIT_CHUNK_BLOCK;
    uint64_t *running = (uint64_t*)scratch;
    uint64_t *block_sums = (uint64_t*)scratch + 2;
    uint64_t cpb;

    if (nblocks > CUMO_BIT_WHERE_MAX_BLOCKS) nblocks = CUMO_BIT_WHERE_MAX_BLOCKS;
    cpb = (nchunks + nblocks - 1) / nblocks;
    cumo_bit_where_partial_kernel<<<nblocks, CUMO_BIT_CHUNK_BLOCK>>>(a,p,s,idx,n,nw,contiguous,0,nchunks,cpb,block_sums);
    cumo_bit_where_scan_kernel<<<1, CUMO_BIT_CHUNK_BLOCK>>>(block_sums,nblocks,running);
    cumo_bit_mask_scatter_kernel<<<nblocks, CUMO_BIT_CHUNK_BLOCK>>>(a,p,s,idx,n,nw,contiguous,nchunks,cpb,out,p2,s2,idx2,block_sums);
    cumo_cuda_runtime_check_kernel_launch();
}
