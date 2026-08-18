__global__ void <%="cumo_#{c_iter}_elementwise_kernel"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a3, size_t p3, ssize_t s3, size_t *idx3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a1, cumo_bit_pos(p1,s1,idx1,i), x);
        CUMO_BIT_DIGIT y = m_<%=name%>(x) & 1u;
        CUMO_STORE_BIT(a3, cumo_bit_pos(p3,s3,idx3,i), y);
    }
}

__global__ void <%="cumo_#{c_iter}_contiguous_kernel"%>(CUMO_BIT_DIGIT *a1, ssize_t o1, uint64_t w1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n, uint64_t w3)
{
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w3; w += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x = cumo_bit_gather_word(a1,o1,w1,w);
        cumo_bit_store_word(a3, w, m_<%=name%>(x), p3, n);
    }
}

void <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a3, size_t p3, ssize_t s3, size_t *idx3, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_elementwise_kernel"%><<<grid_dim, block_dim>>>(a1,p1,s1,idx1,a3,p3,s3,idx3,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n)
{
    ssize_t o1 = (ssize_t)p1 - (ssize_t)p3;
    uint64_t w1 = (p1 + n + CUMO_NB - 1) / CUMO_NB;
    uint64_t w3 = (p3 + n + CUMO_NB - 1) / CUMO_NB;
    size_t grid_dim = cumo_get_grid_dim(w3);
    size_t block_dim = cumo_get_block_dim(w3);
    <%="cumo_#{c_iter}_contiguous_kernel"%><<<grid_dim, block_dim>>>(a1,o1,w1,a3,p3,n,w3);
    cumo_cuda_runtime_check_kernel_launch();
}
