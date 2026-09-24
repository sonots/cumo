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
<%="cumo_#{c_iter}_value"%>(curandStatePhilox4_32_10_t *st, dtype mu, <%= acc_type.empty? ? 'rtype' : acc_type %> sigma)
{
    //<% if is_complex %>
    dtype z;
    CUMO_REAL(z) = cumo_rand_normal(st) * sigma + CUMO_REAL(mu);
    CUMO_IMAG(z) = cumo_rand_normal(st) * sigma + CUMO_IMAG(mu);
    return z;
    //<% elsif !acc_type.empty? %>
    return <%=from_acc%>(cumo_rand_normal(st) * sigma + <%=to_acc%>(mu));
    //<% else %>
    return cumo_rand_normal(st) * sigma + mu;
    //<% end %>
}

<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, cumo_na_indexer_t indexer, uint64_t seed, uint64_t offset, dtype mu, <%= acc_type.empty? ? 'rtype' : acc_type %> sigma)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        curand_init(seed, offset + i, 0, &st);
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = <%="cumo_#{c_iter}_value"%>(&st, mu, sigma);
    }
}
<% end %>
<% end %>

#undef cumo_rand_normal

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

<% layouts.each do |pfx, atype, at| %>
void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, cumo_na_indexer_t* indexer, uint64_t seed, uint64_t offset, dtype mu, <%= acc_type.empty? ? 'rtype' : acc_type %> sigma)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_#{pfx}kernel", "*a1,*indexer,seed,offset,mu,sigma") %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
<% end %>
