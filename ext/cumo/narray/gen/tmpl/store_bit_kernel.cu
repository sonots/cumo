<% unless c_iter.include? 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_index_kernel"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, size_t *idx1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a2, p2 + idx2[i], x);
        *(dtype*)(p1 + idx1[i]) = m_from_real(x);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_index_kernel"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, ssize_t s1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a2, p2 + idx2[i], x);
        *(dtype*)(p1 + (i * s1)) = m_from_real(x);
    }
}

__global__ void <%="cumo_#{c_iter}_index_stride_kernel"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a2, p2 + (i * s2), x);
        *(dtype*)(p1 + idx1[i]) = m_from_real(x);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_stride_kernel"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, ssize_t s1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a2, p2 + (i * s2), x);
        *(dtype*)(p1 + (i * s1)) = m_from_real(x);
    }
}

void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, size_t *idx1, size_t *idx2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_index_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a2,idx1,idx2,n);
}

void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, ssize_t s1, size_t *idx2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_index_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a2,s1,idx2,n);
}

void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a2,idx1,s2,n);
}

void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(char *p1, size_t p2, CUMO_BIT_DIGIT *a2, ssize_t s1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,a2,s1,s2,n);
}

<% end %>
