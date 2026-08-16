<% if type_name == 'robject' || name == 'map' %>
<% else %>

//<% if is_int and name == 'reciprocal' %>
#define cumo_check_intdivzero(x) \
    if ((x)==0) {                \
        *divzero = 1;            \
        continue;                \
    }
//<% else %>
#define cumo_check_intdivzero(x) {}
//<% end %>

__global__ void <%="cumo_#{c_iter}_index_index_kernel"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(*(dtype*)(p1 + idx1[i]));
        *(dtype*)(p2 + idx2[i]) = m_<%=name%>(*(dtype*)(p1 + idx1[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_index_stride_kernel"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(*(dtype*)(p1 + idx1[i]));
        *(dtype*)(p2 + (i * s2)) = m_<%=name%>(*(dtype*)(p1 + idx1[i]));
    }
}

__global__ void <%="cumo_#{c_iter}_stride_index_kernel"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(*(dtype*)(p1 + (i * s1)));
        *(dtype*)(p2 + idx2[i]) = m_<%=name%>(*(dtype*)(p1 + (i * s1)));
    }
}

__global__ void <%="cumo_#{c_iter}_stride_stride_kernel"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(*(dtype*)(p1 + (i * s1)));
        *(dtype*)(p2 + (i * s2)) = m_<%=name%>(*(dtype*)(p1 + (i * s1)));
    }
}

__global__ void <%="cumo_#{c_iter}_contiguous_kernel"%>(char *p1, char *p2, uint64_t n, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        cumo_check_intdivzero(((dtype*)p1)[i]);
        ((dtype*)p2)[i] = m_<%=name%>(((dtype*)p1)[i]);
    }
}

void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_index_kernel"%><<<grid_dim, block_dim>>>(p1,p2,idx1,idx2,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,idx1,s2,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_index_kernel"%><<<grid_dim, block_dim>>>(p1,p2,s1,idx2,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_stride_kernel"%><<<grid_dim, block_dim>>>(p1,p2,s1,s2,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, uint64_t n, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_contiguous_kernel"%><<<grid_dim, block_dim>>>(p1,p2,n,divzero);
    cumo_cuda_runtime_check_kernel_launch();
}
#undef cumo_check_intdivzero
<% end %>
