<% unless type_name == 'robject' %>
__global__ void <%="#{c_iter}_index_kernel"%>(char *ptr, size_t *idx, dtype val, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + idx[i]) = val;
    }
}

__global__ void <%="#{c_iter}_stride_kernel"%>(char*ptr, ssize_t step, dtype val, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + (i*step)) = val;
    }
}

void <%="#{c_iter}_index_kernel_launch"%>(char *ptr, size_t *idx, dtype val, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_index_kernel"%><<<gridDim, blockDim>>>(ptr,idx,val,n);
}

void <%="#{c_iter}_stride_kernel_launch"%>(char *ptr, ssize_t step, dtype val, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(ptr,step,val,n);
}
<% end %>
