<% unless type_name == 'robject' %>

//<% if is_int and %w[divmod].include? name %>
#define cumo_check_intdivzero(y) \
    if ((y)==0) {                \
        *divzero = 1;            \
        continue;                \
    }
//<% else %>
#define cumo_check_intdivzero(y) {}
//<% end %>

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(*(dtype*)(p2+(i*s2)));
        m_<%=name%>(*(dtype*)(p1+(i*s1)),*(dtype*)(p2+(i*s2)),*(dtype*)(p3+(i*s3)), *(dtype*)(p4+(i*s4)));
    }
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,p4,s1,s2,s3,s4,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}
#undef cumo_check_intdivzero
<% end %>
