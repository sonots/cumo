<% unless type_name == 'robject' %>
__global__ void <%="#{c_iter}_contiguous_kernel"%>(char *p1, char *p2, char *p3, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        ((dtype*)p3)[i] = m_<%=name%>(((dtype*)p1)[i],((dtype*)p2)[i]);
    }
}

__global__ void <%="#{c_iter}_stride_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(p3+(i*s3)) = m_<%=name%>(*(dtype*)(p1+(i*s1)),*(dtype*)(p2+(i*s2)));
    }
}

void <%="#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, char *p3, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_contiguous_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,N);
}

void <%="#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,N);
}
<% end %>
