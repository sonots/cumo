<%
if is_int && !is_object
  rand_bit = (/Int64$/ =~ class_name) ? 64 : 32
  rand_type = "uint#{rand_bit}_t"
else
  rand_type = "dtype"
end
%>
<% unless is_object %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#include <curand_kernel.h>

<% if is_double_precision %>
// curand_uniform answers (0,1], and rand is documented as [low,high).
#define cumo_rand_uniform(st) (1.0 - curand_uniform_double(st))
<% else %>
#define cumo_rand_uniform(st) (1.0f - curand_uniform(st))
<% end %>

__device__ static dtype
<%="cumo_#{c_iter}_value"%>(curandStatePhilox4_32_10_t *st, dtype low, <%=rand_type%> max, int shift)
{
    //<% if is_int %>
    <%=rand_type%> x;
    do {
        //<% if rand_bit == 64 %>
        x = curand(st);
        x <<= 32;
        x |= curand(st);
        x >>= shift;
        //<% else %>
        x = curand(st) >> shift;
        //<% end %>
    } while (x >= max);
    return (dtype)x + low;
    //<% elsif is_complex %>
    dtype z;
    CUMO_REAL(z) = cumo_rand_uniform(st) * CUMO_REAL(max) + CUMO_REAL(low);
    CUMO_IMAG(z) = cumo_rand_uniform(st) * CUMO_IMAG(max) + CUMO_IMAG(low);
    return z;
    //<% else %>
    return (dtype)(cumo_rand_uniform(st) * max) + low;
    //<% end %>
}

__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        // Keying on the element rather than the thread keeps the result the same
        // whatever grid the launch happens to pick.
        curand_init(seed, offset + i, 0, &st);
        *(dtype*)(p1+idx1[i]) = <%="cumo_#{c_iter}_value"%>(&st, low, max, shift);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, offset + i, 0, &st);
        *(dtype*)(p1+(i*s1)) = <%="cumo_#{c_iter}_value"%>(&st, low, max, shift);
    }
}

#undef cumo_rand_uniform

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<grid_dim, block_dim>>>(p1,idx1,seed,offset,low,max,shift,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, int shift, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,s1,seed,offset,low,max,shift,n);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
