<% unless type_name == 'robject' %>
// A Ruby numeric operand rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. Either side of a
// module function can be the numeric one, so use_scalar says which side sv is
// on. It is the same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1+(i*s1));
        dtype y;
        switch (use_scalar) {
        case 1:  y = sv; break;
        case 2:  y = x; x = sv; break;
        default: y = *(dtype*)(p2+(i*s2));
        }
        *(dtype*)(p3+(i*s3)) = m_<%=name%>(x,y);
    }
}

static void <%="cumo_#{c_iter}_stride_kernel_dispatch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n,sv,use_scalar);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_stride_kernel_dispatch"%>(p1,p2,p3,s1,s2,s3,n,sv,0);
}

void <%="cumo_#{c_iter}_s_stride_kernel_launch"%>(char *p1, char *p3, ssize_t s1, ssize_t s3, uint64_t n, dtype sv, int scalar_is_left)
{
    <%="cumo_#{c_iter}_stride_kernel_dispatch"%>(p1,NULL,p3,s1,0,s3,n,sv,scalar_is_left ? 2 : 1);
}
<% end %>
