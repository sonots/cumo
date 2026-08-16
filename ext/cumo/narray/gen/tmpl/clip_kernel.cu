<% unless type_name == 'robject' %>
__global__ void <%="cumo_#{c_iter}_kernel"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n, int* invalid)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype min = *(dtype*)(p2+(i*s2));
        dtype max = *(dtype*)(p3+(i*s3));
        // Leaving the element unwritten keeps a scalar min > max, where every
        // element takes this branch, from touching the output before it raises.
        if (m_gt(min,max)) { *invalid = 1; continue; }
        if (m_lt(x,min)) { x = min; }
        if (m_gt(x,max)) { x = max; }
        *(dtype*)(p4+(i*s4)) = x;
    }
}

void <%="cumo_#{c_iter}_kernel_launch"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n, int* invalid)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,p4,s1,s2,s3,s4,n,invalid);
    cumo_cuda_runtime_check_kernel_launch();
}

__global__ void <%="cumo_#{c_iter}_min_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype min = *(dtype*)(p2+(i*s2));
        *(dtype*)(p3+(i*s3)) = m_lt(x,min) ? min : x;
    }
}

void <%="cumo_#{c_iter}_min_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_min_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
}

__global__ void <%="cumo_#{c_iter}_max_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype max = *(dtype*)(p2+(i*s2));
        *(dtype*)(p3+(i*s3)) = m_gt(x,max) ? max : x;
    }
}

void <%="cumo_#{c_iter}_max_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_max_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
