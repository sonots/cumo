<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        m_<%=name%>(*(dtype*)(p1+(i*s1)),*(dtype*)(p2+(i*s2)),*(dtype*)(p3+(i*s3)), *(dtype*)(p4+(i*s4)));
    }
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,p4,s1,s2,s3,s4,n);
}
<% end %>
