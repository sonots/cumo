// atomicAdd on double arrived with sm_60. Before that the compare-and-swap loop
// the CUDA programming guide gives is the only way to get one, and the default
// target of the CUDA releases cumo builds against is still below it.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ static inline void <%=c_iter%>_add_double(double *address, double val)
{
    unsigned long long int *p = (unsigned long long int*)address;
    unsigned long long int old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}
#else
__device__ static inline void <%=c_iter%>_add_double(double *address, double val) { atomicAdd(address, val); }
#endif

// A kernel cannot raise, so an item that does not fit the output is reported
// through the flag and the value is left for the message. Any offending item
// will do, so the two stores race harmlessly.
#define cumo_check_bincount_item(x, n)  \
    if ((x) >= (n)) {                   \
        *item = (x);                    \
        *oor = 1;                       \
        continue;                       \
    }

<%
[["32","u_int32_t","unsigned int","1u"],
 ["64","u_int64_t","unsigned long long","1ull"]].each do |bits,cnt_type,atom_type,one|
%>
__global__ void <%="cumo_#{c_iter}_#{bits}_init_kernel"%>(char *p2, ssize_t s2, uint64_t n)
{
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += blockDim.x * gridDim.x) {
        *(<%=cnt_type%>*)(p2 + s2 * x) = 0;
    }
}

__global__ void <%="cumo_#{c_iter}_#{bits}_kernel"%>(char *p1, ssize_t s1, size_t *idx1, uint64_t count, char *p2, ssize_t s2, uint64_t n, int *oor, size_t *item)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(idx1 ? (p1 + idx1[i]) : (p1 + s1 * i)));
        cumo_check_bincount_item(x, n);
        atomicAdd((<%=atom_type%>*)(p2 + s2 * x), <%=one%>);
    }
}

void <%="cumo_#{c_iter}_#{bits}_kernel_launch"%>(char *p1, ssize_t s1, size_t *idx1, uint64_t count, char *p2, ssize_t s2, uint64_t n, int *oor, size_t *item)
{
    <%="cumo_#{c_iter}_#{bits}_init_kernel"%><<<cumo_get_grid_dim(n), cumo_get_block_dim(n)>>>(p2,s2,n);
    cumo_cuda_runtime_check_kernel_launch();
    <%="cumo_#{c_iter}_#{bits}_kernel"%><<<cumo_get_grid_dim(count), cumo_get_block_dim(count)>>>(p1,s1,idx1,count,p2,s2,n,oor,item);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>

<%
[["sf","float","atomicAdd"],
 ["df","double","#{c_iter}_add_double"]].each do |fn,cnt_type,add|
%>
__global__ void <%="cumo_#{c_iter}_#{fn}_init_kernel"%>(char *p3, ssize_t s3, uint64_t n)
{
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < n; x += blockDim.x * gridDim.x) {
        *(<%=cnt_type%>*)(p3 + s3 * x) = 0;
    }
}

__global__ void <%="cumo_#{c_iter}_#{fn}_kernel"%>(char *p1, ssize_t s1, char *p2, ssize_t s2, uint64_t count, char *p3, ssize_t s3, uint64_t n, int *oor, size_t *item)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
        size_t x = (size_t)(*(dtype*)(p1 + s1 * i));
        cumo_check_bincount_item(x, n);
        <%=add%>((<%=cnt_type%>*)(p3 + s3 * x), *(<%=cnt_type%>*)(p2 + s2 * i));
    }
}

void <%="cumo_#{c_iter}_#{fn}_kernel_launch"%>(char *p1, ssize_t s1, char *p2, ssize_t s2, uint64_t count, char *p3, ssize_t s3, uint64_t n, int *oor, size_t *item)
{
    <%="cumo_#{c_iter}_#{fn}_init_kernel"%><<<cumo_get_grid_dim(n), cumo_get_block_dim(n)>>>(p3,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
    <%="cumo_#{c_iter}_#{fn}_kernel"%><<<cumo_get_grid_dim(count), cumo_get_block_dim(count)>>>(p1,s1,p2,s2,count,p3,s3,n,oor,item);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>

#undef cumo_check_bincount_item
