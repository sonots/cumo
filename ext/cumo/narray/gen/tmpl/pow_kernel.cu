<% unless type_name == 'robject' %>
__global__ void <%="#{c_iter}_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p3 + (i * s3)) = m_pow(*(dtype*)(p1 + (i * s1)), *(dtype*)(p2 + (i * s2)));
    }
}

__global__ void <%="#{c_iter}_int32_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p3 + (i * s3)) = m_pow_int(*(dtype*)(p1 + (i * s1)), *(int32_t*)(p2 + (i * s2)));
    }
}

void <%="#{c_iter}_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,n);
}

void <%="#{c_iter}_int32_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_int32_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,n);
}
<% end %>
