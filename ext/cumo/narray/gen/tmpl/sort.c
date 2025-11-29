<% (is_float ? ["_ignan","_prnan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    size_t n;
    char *ptr;
    ssize_t step;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR(lp, 0, ptr, step);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    <%=type_name%>_qsort<%=j%>(ptr, n, step);
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  If true, propagete NaN. If false, ignore NaN.
<% else %>
  @overload <%=name%>(axis:nil)
<% end %>
  @param [Numeric,Array,Range] axis  Performs <%=name%> along the axis.
  @return [Cumo::<%=class_name%>] returns result of <%=name%>.
  @example
      Cumo::DFloat[3,4,1,2].sort # => Cumo::DFloat[1,2,3,4]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_t ndf = {0, CUMO_NDF_HAS_LOOP|CUMO_NDF_FLAT_REDUCE, 2,0, ain,0};

    if (!CUMO_TEST_INPLACE(self)) {
        self = cumo_na_copy(self);
    }
  <% if is_float %>
    ndf.func = <%=c_iter%>_ignan;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_prnan);
  <% else %>
    ndf.func = <%=c_iter%>;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    cumo_na_ndloop(&ndf, 2, self, reduce);
    return self;
}
