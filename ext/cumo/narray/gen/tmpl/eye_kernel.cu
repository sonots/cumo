<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char*ptr, ssize_t s0, ssize_t s1, ssize_t kofs, dtype data, uint64_t n0, uint64_t n1, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        uint64_t i0 = i / n1;
        uint64_t i1 = i - (i0 * n1);
        *(dtype*)(ptr + (i0*s0) + (i1*s1)) = (i0+kofs==i1) ? data : m_zero;
    }
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *ptr, ssize_t s0, ssize_t s1, ssize_t kofs, dtype data, uint64_t n0, uint64_t n1)
{
    uint64_t n = n0 * n1;
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(ptr,s0,s1,kofs,data,n0,n1,n);
}
<% end %>

