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
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(ptr,s0,s1,kofs,data,n0,n1,n);
}
<% end %>

