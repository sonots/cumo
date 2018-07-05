<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *ptr, size_t *idx, dtype val, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + idx[i]) = val;
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char*ptr, ssize_t step, dtype val, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(ptr + (i*step)) = val;
    }
}

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *ptr, size_t *idx, dtype val, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<grid_dim, block_dim>>>(ptr,idx,val,n);
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *ptr, ssize_t step, dtype val, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(ptr,step,val,n);
}
<% end %>
