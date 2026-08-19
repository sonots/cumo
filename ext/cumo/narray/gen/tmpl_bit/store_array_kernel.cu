// Bit i of the packed buffer the host filled
__device__ static inline CUMO_BIT_DIGIT
<%="cumo_#{c_iter}_bit"%>(const CUMO_BIT_DIGIT* z, uint64_t i)
{
    return (z[i / CUMO_NB] >> (i % CUMO_NB)) & 1u;
}

__global__ void <%="cumo_#{c_iter}_index_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, const CUMO_BIT_DIGIT* z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT y = <%="cumo_#{c_iter}_bit"%>(z, i);
        CUMO_STORE_BIT(a1, p1 + idx1[i], y);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, const CUMO_BIT_DIGIT* z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT y = <%="cumo_#{c_iter}_bit"%>(z, i);
        CUMO_STORE_BIT(a1, p1 + s1 * i, y);
    }
}

// The staging buffer starts at bit 0, so the destination's own offset is the
// only shift between the two, and the words at either end of the run are
// shared with elements the store must not touch.
__global__ void <%="cumo_#{c_iter}_contiguous_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, const CUMO_BIT_DIGIT* z, uint64_t nz, uint64_t n, uint64_t w1)
{
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w1; w += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x = cumo_bit_gather_word(z, -(ssize_t)p1, nz, w);
        cumo_bit_store_word(a1, w, x, p1, n);
    }
}

__global__ void <%="cumo_#{c_iter}_index_scalar_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, CUMO_BIT_DIGIT z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_STORE_BIT(a1, p1 + idx1[i], z);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_scalar_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, CUMO_BIT_DIGIT z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_STORE_BIT(a1, p1 + s1 * i, z);
    }
}

void <%="cumo_#{c_iter}_index_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, CUMO_BIT_DIGIT* z, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<grid_dim, block_dim>>>(a1,p1,idx1,z,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, CUMO_BIT_DIGIT* z, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(a1,p1,s1,z,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT* z, uint64_t n)
{
    uint64_t nz = (n + CUMO_NB - 1) / CUMO_NB;
    uint64_t w1 = (p1 + n + CUMO_NB - 1) / CUMO_NB;
    size_t grid_dim = cumo_get_grid_dim(w1);
    size_t block_dim = cumo_get_block_dim(w1);
    <%="cumo_#{c_iter}_contiguous_kernel"%><<<grid_dim, block_dim>>>(a1,p1,z,nz,n,w1);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, size_t *idx1, CUMO_BIT_DIGIT z, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_scalar_kernel"%><<<grid_dim, block_dim>>>(a1,p1,idx1,z,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, CUMO_BIT_DIGIT z, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_scalar_kernel"%><<<grid_dim, block_dim>>>(a1,p1,s1,z,n);
    cumo_cuda_runtime_check_kernel_launch();
}
