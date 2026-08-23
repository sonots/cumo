#undef int_t
#define int_t unsigned long long int

// One thread folds a whole word of bits at a time and one thread per block
// reaches the accumulator, instead of one thread per bit each adding its own.
__global__ void <%="cumo_#{c_iter}_chunk_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, char *p2, uint64_t n, uint64_t nw, uint64_t nc, int contiguous)
{
    // m_<%=name%>(0) is 1 exactly when the zeros are what this method counts,
    // which is the complement cumo_bit_chunk takes.
    const int invert = m_<%=name%>(0);
    uint64_t cnt = 0, total;

    for (uint64_t c = blockIdx.x * blockDim.x + threadIdx.x; c < nc; c += blockDim.x * gridDim.x) {
        cnt += (uint64_t)__popc(cumo_bit_chunk(a1, p1, s1, idx1, n, nw, c, contiguous, invert));
    }
    cumo_bit_block_exscan(cnt, &total);
    if (threadIdx.x == 0 && total != 0) {
        atomicAdd((int_t*)p2, (int_t)total);
    }
}

__global__ void <%="cumo_#{c_iter}_index_stride_kernel"%>(size_t p1, char* p2, CUMO_BIT_DIGIT *a1, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x=0;
        CUMO_LOAD_BIT(a1, p1 + idx1[i], x);
        if (m_<%=name%>(x)) {
            atomicAdd((int_t*)(p2 + i * s2), (int_t)1);
        }
    }
}

__global__ void <%="cumo_#{c_iter}_stride_stride_kernel"%>(size_t p1, char* p2, CUMO_BIT_DIGIT *a1, ssize_t s1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x=0;
        CUMO_LOAD_BIT(a1, p1 + i * s1, x);
        if (m_<%=name%>(x)) {
            atomicAdd((int_t*)(p2 + i * s2), (int_t)1);
        }
    }
}

void <%="cumo_#{c_iter}_chunk_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, char *p2, uint64_t n)
{
    uint64_t nc = (n + CUMO_NB - 1) / CUMO_NB;
    int contiguous = (idx1 == NULL && s1 == 1);
    uint64_t nw = contiguous ? (p1 + n + CUMO_NB - 1) / CUMO_NB : 0;
    uint64_t nblocks = (nc + CUMO_BIT_CHUNK_BLOCK - 1) / CUMO_BIT_CHUNK_BLOCK;

    if (nc == 0) return;
    if (nblocks > CUMO_MAX_GRID_DIM) nblocks = CUMO_MAX_GRID_DIM;
    <%="cumo_#{c_iter}_chunk_kernel"%><<<nblocks, CUMO_BIT_CHUNK_BLOCK>>>(a1,p1,s1,idx1,p2,n,nw,nc,contiguous);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(size_t p1, char *p2, CUMO_BIT_DIGIT *a1, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a1,idx1,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(size_t p1, char *p2, CUMO_BIT_DIGIT *a1, ssize_t s1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a1,s1,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
}

#undef int_t
