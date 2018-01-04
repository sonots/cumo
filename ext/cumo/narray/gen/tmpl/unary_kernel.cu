<% if c_iter.include?('robject') || name == 'map' %>
<% else %>
__global__ void <%="#{c_iter}_index_index_kernel"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(p2 + idx2[i]) = m_<%=name%>(*(dtype*)(p1 + idx1[i]));
    }
}

__global__ void <%="#{c_iter}_index_stride_kernel"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(p2 + (i * s2)) = m_<%=name%>(*(dtype*)(p1 + idx1[i]));
    }
}

__global__ void <%="#{c_iter}_stride_index_kernel"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(p2 + idx2[i]) = m_<%=name%>(*(dtype*)(p1 + (i * s1)));
    }
}

__global__ void <%="#{c_iter}_stride_stride_kernel"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        *(dtype*)(p2 + (i * s2)) = m_<%=name%>(*(dtype*)(p1 + (i * s1)));
    }
}

__global__ void <%="#{c_iter}_contiguous_kernel"%>(char *p1, char *p2, size_t N)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        ((dtype*)p2)[i] = m_<%=name%>(((dtype*)p1)[i]);
    }
}

void <%="#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_index_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,idx2,N);
}

void <%="#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_index_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,s2,N);
}

void <%="#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_stride_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,s1,idx2,N);
}

void <%="#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_stride_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,s1,s2,N);
}

void <%="#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, size_t N)
{
    size_t gridDim = get_gridDim(N);
    size_t blockDim = get_blockDim(N);
    <%="#{c_iter}_contiguous_kernel"%><<<gridDim, blockDim>>>(p1,p2,N);
}
<% end %>
