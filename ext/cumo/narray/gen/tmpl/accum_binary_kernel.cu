<% unless defined?($cumo_narray_gen_tmpl_accum_binary_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_binary_kernel_included = 1 %>

<% unless type_name == 'robject' %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

// The product is never an array of its own: MapIn takes one element of each
// operand, which is what a zip reduction hands it. The accumulator stays dtype
// rather than widening, as the host loop it replaces did.
struct <%="cumo_#{type_name}_#{name}#{nan}_impl"%> {
    __device__ dtype Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype x, dtype y, int64_t /*index*/) {
        dtype z = m_zero;
        m_<%=name%><%=nan%>(x, y, z);
        return z;
    }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_add(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
};
//<% end %>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>
void <%="cumo_#{type_name}_#{name}#{nan}_kernel_launch"%>(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* in2)
{
    cumo_reduce_zip_split<dtype, dtype, <%="cumo_#{type_name}_#{name}#{nan}_impl"%>>(*arg, *in2, <%="cumo_#{type_name}_#{name}#{nan}_impl"%>{});
}
//<% end %>
<% end %>
<% end %>
