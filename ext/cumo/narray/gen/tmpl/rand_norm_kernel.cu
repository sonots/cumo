<% unless is_object %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

#include <curand_kernel.h>

<% if is_double_precision %>
#define cumo_rand_normal(st) curand_normal_double(st)
<% else %>
#define cumo_rand_normal(st) curand_normal(st)
<% end %>

__device__ static dtype
<%="cumo_#{c_iter}_value"%>(curandStatePhilox4_32_10_t *st, dtype mu, rtype sigma)
{
    //<% if is_complex %>
    dtype z;
    CUMO_REAL(z) = cumo_rand_normal(st) * sigma + CUMO_REAL(mu);
    CUMO_IMAG(z) = cumo_rand_normal(st) * sigma + CUMO_IMAG(mu);
    return z;
    //<% else %>
    return cumo_rand_normal(st) * sigma + mu;
    //<% end %>
}

__global__ void <%="cumo_#{c_iter}_index_kernel"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, offset + i, 0, &st);
        *(dtype*)(p1+idx1[i]) = <%="cumo_#{c_iter}_value"%>(&st, mu, sigma);
    }
}

__global__ void <%="cumo_#{c_iter}_stride_kernel"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, offset + i, 0, &st);
        *(dtype*)(p1+(i*s1)) = <%="cumo_#{c_iter}_value"%>(&st, mu, sigma);
    }
}

#undef cumo_rand_normal

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, size_t *idx1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_index_kernel"%><<<grid_dim, block_dim>>>(p1,idx1,seed,offset,mu,sigma,n);
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, ssize_t s1, uint64_t seed, uint64_t offset, dtype mu, rtype sigma, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{c_iter}_stride_kernel"%><<<grid_dim, block_dim>>>(p1,s1,seed,offset,mu,sigma,n);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
