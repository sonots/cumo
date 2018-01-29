<% unless c_iter.include? 'robject' %>
__global__ void <%="#{c_iter}_index_index_kernel"%>(char *p1, size_t p2, BIT_DIGIT *a2, size_t *idx1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        BIT_DIGIT x;
        LOAD_BIT(a2, p2 + idx2[i], x);
        *(dtype*)(p1 + idx1[i]) = m_from_real(x);
    }
}

__global__ void <%="#{c_iter}_stride_index_kernel"%>(char *p1, size_t p2, BIT_DIGIT *a2, ssize_t s1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        BIT_DIGIT x;
        LOAD_BIT(a2, p2 + idx2[i], x);
        *(dtype*)(p1 + (i * s1)) = m_from_real(x);
    }
}

__global__ void <%="#{c_iter}_index_stride_kernel"%>(char *p1, size_t p2, BIT_DIGIT *a2, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        BIT_DIGIT x;
        LOAD_BIT(a2, p2 + (i * s2), x);
        *(dtype*)(p1 + idx1[i]) = m_from_real(x);
    }
}

__global__ void <%="#{c_iter}_stride_stride_kernel"%>(char *p1, size_t p2, BIT_DIGIT *a2, ssize_t s1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        BIT_DIGIT x;
        LOAD_BIT(a2, p2 + (i * s2), x);
        *(dtype*)(p1 + (i * s1)) = m_from_real(x);
    }
}

void <%="#{c_iter}_index_index_kernel_launch"%>(char *p1, size_t p2, BIT_DIGIT *a2, size_t *idx1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_index_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,a2,idx1,idx2,n);
}

void <%="#{c_iter}_stride_index_kernel_launch"%>(char *p1, size_t p2, BIT_DIGIT *a2, ssize_t s1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_stride_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,a2,s1,idx2,n);
}

void <%="#{c_iter}_index_stride_kernel_launch"%>(char *p1, size_t p2, BIT_DIGIT *a2, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_index_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,a2,idx1,s2,n);
}

void <%="#{c_iter}_stride_stride_kernel_launch"%>(char *p1, size_t p2, BIT_DIGIT *a2, ssize_t s1, ssize_t s2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_stride_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,a2,s1,s2,n);
}

<% end %>
