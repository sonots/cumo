<% (is_float ? ["","_nan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2;
    ssize_t  s1, s2;
    dtype    x, y;

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    //printf("i=%lu p1=%lx s1=%lu p2=%lx s2=%lu\n",i,(size_t)p1,s1,(size_t)p2,s2);

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=j%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
    CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
    //printf("i=%lu x=%f\n",i,x);
    for (i--; i--;) {
        CUMO_GET_DATA_STRIDE(p1,s1,dtype,y);
        m_<%=name%><%=j%>(x,y);
        CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
        //printf("i=%lu x=%f\n",i,x);
    }
}
<% end %>

/*
  <%=name%> of self.
  @overload <%=name%>(axis:nil, nan:false)
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN if exists).
  @return [Cumo::<%=class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_CUM,
                     2, 1, ain, aout };

  <% if is_float %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_nan);
  <% else %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    return cumo_na_ndloop(&ndf, 2, self, reduce);
}
