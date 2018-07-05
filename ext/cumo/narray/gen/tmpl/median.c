<% (is_float ? ["_ignan","_prnan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    size_t n;
    char *p1, *p2;
    dtype *buf;

    CUMO_INIT_COUNTER(lp, n);
    p1 = (lp->args[0]).ptr + (lp->args[0].iter[0]).pos;
    p2 = (lp->args[1]).ptr + (lp->args[1].iter[0]).pos;
    buf = (dtype*)p1;

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=j%>", "<%=type_name%>");
    <%=type_name%>_qsort<%=j%>(buf, n, sizeof(dtype));

    <% if is_float %>
    for (; n; n--) {
        if (!isnan(buf[n-1])) break;
    }
    <% end %>

    if (n==0) {
        *(dtype*)p2 = buf[0];
    }
    else if (n%2==0) {
        *(dtype*)p2 = (buf[n/2-1]+buf[n/2])/2;
    }
    else {
        *(dtype*)p2 = buf[(n-1)/2];
    }
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan (keyword) If true, propagete NaN. If false, ignore NaN.
<% else %>
  @overload <%=name%>(axis:nil, keepdims:false)
<% end %>
  @param [Numeric,Array,Range] axis (keyword) Affected dimensions.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::<%=class_name%>] returns <%=name%> of self.
*/

static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{INT2FIX(0),0}};
    cumo_ndfunc_t ndf = {0, CUMO_NDF_HAS_LOOP|CUMO_NDF_FLAT_REDUCE, 2,1, ain,aout};

    self = cumo_na_copy(self); // as temporary buffer
  <% if is_float %>
    ndf.func = <%=c_iter%>_ignan;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_prnan);
  <% else %>
    ndf.func = <%=c_iter%>;
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    v = cumo_na_ndloop(&ndf, 2, self, reduce);
    return <%=type_name%>_extract(v);
}
