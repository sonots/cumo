<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1 + idx1[i]);
        CUMO_BIT_DIGIT b = (m_<%=name%>(x)) ? 1:0;
        CUMO_STORE_BIT(a2,p2+(i*s2),b);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        CUMO_BIT_DIGIT b = (m_<%=name%>(x)) ? 1:0;
        CUMO_STORE_BIT(a2,p2+(i*s2),b);
    }
}

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<grid_dim, block_dim>>>(p1,a2,p2,idx1,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s1, ssize_t s2, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,a2,p2,s1,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
