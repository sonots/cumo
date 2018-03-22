<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(size_t n, char *p1, ssize_t s1, char *p2);
<% end %>

static void
<%=c_iter%><%=nan%>(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2;
    ssize_t  s1;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    p2 = lp->args[1].ptr + lp->args[1].iter[0].pos;

    // TODO(sonots): How to compute Kahan summation algorithm in parallel?
    // TODO(sonots): Implement nan CUDA version
    <% if type_name == 'robject' || name == 'kahan_sum' || nan == '_nan' %>
    SHOW_CPU_WARNING_ONCE("<%=name%><%=nan%>", "<%=type_name%>");
    *(<%=dtype%>*)p2 = f_<%=name%><%=nan%>(n,p1,s1);
    <% else %>
    cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(n,p1,s1,p2);
    <% end %>
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (avoid NaN for sum/mean etc, or, return NaN for min/max etc).
<% else %>
  @overload <%=name%>(axis:nil, keepdims:false)
<% end %>
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::<%=class_name%>] returns result of <%=name%>.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    ndfunc_arg_in_t ain[2] = {{cT,0},{sym_reduce,0}};
    ndfunc_arg_out_t aout[1] = {{<%=result_class%>,0}};
    ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP_NIP|NDF_FLAT_REDUCE, 2, 1, ain, aout };

  <% if is_float %>
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_nan);
  <% else %>
    reduce = na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    v =  na_ndloop(&ndf, 2, self, reduce);
  <% if result_class == "cT" %>
    return <%=type_name%>_extract(v);
  <% else %>
    return rb_funcall(v,rb_intern("extract"),0);
  <% end %>
}
