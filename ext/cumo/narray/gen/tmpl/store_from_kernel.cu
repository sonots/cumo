<% unless c_iter.include? 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_index_kernel"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = <%=macro%>(*(<%=dtype%>*)(p2 + idx2[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_stride_index_kernel"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + (i * s1)) = <%=macro%>(*(<%=dtype%>*)(p2 + idx2[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_index_stride_kernel"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = <%=macro%>(*(<%=dtype%>*)(p2 + (i * s2)));
    }
}

__global__ void <%="cumo_#{c_iter}_stride_stride_kernel"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + (i * s1)) = <%=macro%>(*(<%=dtype%>*)(p2 + (i * s2)));
    }
}

void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,idx2,n);
}

void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_index_kernel"%><<<gridDim, blockDim>>>(p1,p2,s1,idx2,n);
}

void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,idx1,s2,n);
}

void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_stride_kernel"%><<<gridDim, blockDim>>>(p1,p2,s1,s2,n);
}

<% end %>

