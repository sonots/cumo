<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, char *p2, BIT_DIGIT *a3, size_t p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype y = *(dtype*)(p2+(i*s2));
        BIT_DIGIT b = (m_<%=name%>(x,y)) ? 1:0;
        CUMO_STORE_BIT(a3,p3+(i*s3),b);
    }
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, BIT_DIGIT *a3, size_t p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,a3,p3,s1,s2,s3,n);
}
<% end %>
