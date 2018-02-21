<% unless defined?($cumo_narray_gen_tmpl_gemv_kernel_included) %>
<% $cumo_narray_gen_tmpl_gemv_kernel_included = 1 %>

<% unless type_name == 'robject' %>
//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

__global__ void <%="#{type_name}_gemv#{nan}_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    //for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    //    m_<%=name%><%=nan%>(*(dtype*)(p1+(i*s1)), *(dtype*)(p2+(i*s2)), *(dtype*)(p3+(i*s3)));
    //}
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="#{type_name}_gemv#{nan}_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    //<%="#{type_name}_gemv#{nan}_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,n);
}

//<% end %>
<% end %>
<% end %>
