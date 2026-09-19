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
//<% ([nil] + from_types).each do |from| %>
//<% sfx = from ? "_from_#{from[:id]}" : "" %>

// The product is never an array of its own: MapIn takes one element of each
// operand, which is what a zip reduction hands it. The accumulator stays dtype
// rather than widening, as the host loop it replaces did, except for the
// types that set an accumulator of their own.
<% if from %>
// This one reads the second operand as <%=from[:name]%> and converts each
// element, so an operand of that type never has to be cast into an array first.
<% end %>
struct <%="cumo_#{type_name}_#{name}#{nan}#{sfx}_impl"%> {
<% unless acc_type.empty? %>
    __device__ <%=acc_type%> Identity(int64_t /*index*/) { return <%=acc_zero%>; }
    __device__ <%=acc_type%> MapIn(dtype x, <%= from ? "#{from[:ctype]} y_in" : "dtype y" %>, int64_t /*index*/) {
<% if from %>
        dtype y = <%=from[:macro]%>(y_in);
<% end %>
<% if nan == '_nan' %>
        if (!not_nan(x) || !not_nan(y)) { return <%=acc_zero%>; }
<% end %>
        return <%=to_acc%>(x) * <%=to_acc%>(y);
    }
    __device__ void Reduce(<%=acc_type%> next, <%=acc_type%>& accum) { accum = next + accum; }
    __device__ dtype MapOut(<%=acc_type%> accum) { return <%=from_acc%>(accum); }
<% else %>
    __device__ dtype Identity(int64_t /*index*/) { return m_zero; }
    __device__ dtype MapIn(dtype x, <%= from ? "#{from[:ctype]} y_in" : "dtype y" %>, int64_t /*index*/) {
        dtype z = m_zero;
<% if from %>
        dtype y = <%=from[:macro]%>(y_in);
<% end %>
        m_<%=name%><%=nan%>(x, y, z);
        return z;
    }
    __device__ void Reduce(dtype next, dtype& accum) { accum = m_add(next, accum); }
    __device__ dtype MapOut(dtype accum) { return accum; }
<% end %>
};
//<% end %>
//<% end %>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>
//<% ([nil] + from_types).each do |from| %>
//<% sfx = from ? "_from_#{from[:id]}" : "" %>
void <%="cumo_#{type_name}_#{name}#{nan}#{sfx}_kernel_launch"%>(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* in2)
{
    cumo_reduce_zip_split<dtype, <%= from ? from[:ctype] : "dtype" %>, dtype, <%="cumo_#{type_name}_#{name}#{nan}#{sfx}_impl"%>>(*arg, *in2, <%="cumo_#{type_name}_#{name}#{nan}#{sfx}_impl"%>{});
}
//<% end %>
//<% end %>
<% end %>
<% end %>
