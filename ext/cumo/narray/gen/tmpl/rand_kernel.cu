<%
if is_int && !is_object
  rand_bit = (/Int64$/ =~ class_name) ? 64 : 32
  rand_type = "uint#{rand_bit}_t"
elsif !acc_type.empty?
  rand_type = acc_type
else
  rand_type = "dtype"
end
# The type the caller read the upper bound in, which rand_opt_t carries.
high_type = acc_type.empty? ? "dtype" : acc_type
next_below = is_double_precision ? "nextafter" : "nextafterf"
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

<% if !is_int && acc_type.empty? %>
// The product and the sum round to the nearest value the type holds, which
// can be the upper bound rand does not include, so a draw that reaches it
// takes the neighbour below. A range that does not rise is left as it was.
__device__ static rtype
<%="cumo_#{c_iter}_component"%>(curandStatePhilox4_32_10_t *st, rtype low, rtype max, rtype high)
{
    rtype v = cumo_rand_uniform(st) * max + low;
    if (low < high && v >= high) {
        v = <%=next_below%>(high, low);
    }
    return v;
}
<% end %>

__device__ static dtype
<%="cumo_#{c_iter}_value"%>(curandStatePhilox4_32_10_t *st, dtype low, <%=rand_type%> max, <%=high_type%> high, int shift)
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
    CUMO_REAL(z) = <%="cumo_#{c_iter}_component"%>(st, CUMO_REAL(low), CUMO_REAL(max), CUMO_REAL(high));
    CUMO_IMAG(z) = <%="cumo_#{c_iter}_component"%>(st, CUMO_IMAG(low), CUMO_IMAG(max), CUMO_IMAG(high));
    return z;
    //<% elsif !acc_type.empty? %>
    // Rounding the float into the element type can land on or past the upper
    // bound, which rand does not include, so such a draw takes the largest
    // value the type holds below it. A range that does not rise is left as it
    // was.
    <%=acc_type%> flow = <%=to_acc%>(low);
    dtype h = <%=from_acc%>(cumo_rand_uniform(st) * max + flow);
    if (flow < high && <%=to_acc%>(h) >= high) {
        h = <%=from_acc%>(high);
        if (<%=to_acc%>(h) >= high) {
            h = <%=step_down%>(h);
        }
    }
    return h;
    //<% else %>
    return <%="cumo_#{c_iter}_component"%>(st, low, max, high);
    //<% end %>
}

<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% indexer_dims(pfx.empty?).each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, cumo_na_indexer_t indexer, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, <%=high_type%> high, int shift)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        curandStatePhilox4_32_10_t st;
        // Keying on the element rather than the thread keeps the result the same
        // whatever grid the launch happens to pick.
        curand_init(seed, offset + i, 0, &st);
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = <%="cumo_#{c_iter}_value"%>(&st, low, max, high, shift);
    }
}
<% end %>
<% end %>

#undef cumo_rand_uniform

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

<% layouts.each do |pfx, atype, at| %>
void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, cumo_na_indexer_t* indexer, uint64_t seed, uint64_t offset, dtype low, <%=rand_type%> max, <%=high_type%> high, int shift)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_#{pfx}kernel", "*a1,*indexer,seed,offset,low,max,high,shift", narrow: (%w[a1] if pfx.empty?)) %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
<% end %>
