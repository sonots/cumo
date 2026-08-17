__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype y, z;
        m_<%=name%>(x,y,z);
        *(dtype*)(p2+(i*s2)) = y;
        *(dtype*)(p3+(i*s3)) = z;
    }
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
}
