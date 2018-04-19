<% unless c_iter.include? 'robject' %>
__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *p1, size_t *idx1, dtype* z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = z[i];
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, ssize_t s1, dtype* z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + (i * s1)) = z[i];
    }
}

__global__ void <%="cumo_#{c_iter}_index_scalar_kernel"%>(char *p1, size_t *idx1, dtype z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + idx1[i]) = z;
    }
}

__global__ void <%="cumo_#{c_iter}_stride_scalar_kernel"%>(char *p1, ssize_t s1, dtype z, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *(dtype*)(p1 + (i * s1)) = z;
    }
}

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, dtype* z, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<gridDim, blockDim>>>(p1,idx1,z,n);
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, dtype* z, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<gridDim, blockDim>>>(p1,s1,z,n);
}

void <%="cumo_#{c_iter}_index_scalar_kernel_launch"%>(char *p1, size_t *idx1, dtype z, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_index_scalar_kernel"%><<<gridDim, blockDim>>>(p1,idx1,z,n);
}

void <%="cumo_#{c_iter}_stride_scalar_kernel_launch"%>(char *p1, ssize_t s1, dtype z, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{c_iter}_stride_scalar_kernel"%><<<gridDim, blockDim>>>(p1,s1,z,n);
}

<% end %>
