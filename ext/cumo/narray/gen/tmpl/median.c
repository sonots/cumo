void cumo_<%=type_name%>_median_kernel_launch(cumo_na_reduction_arg_t* arg, int flat, int prnan);

// The rows are sorted with one cub call and the middle of each is picked by a
// kernel of its own. The sort keys put every NaN last, so nan:true only has to
// notice that one is there.
<% (is_float ? ["_ignan","_prnan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    cumo_na_reduction_arg_t arg = cumo_na_make_reduction_arg(lp, 1);
    ssize_t expect = sizeof(dtype);
    int i, flat = 1;

    // Rows have to be laid out end to end for a segmented sort to address them.
    for (i = arg.in_indexer.ndim; --i >= 0;) {
        if (arg.in.step[i] != expect) {
            flat = 0;
            break;
        }
        expect *= (ssize_t)arg.in_indexer.shape[i];
    }

    cumo_<%=type_name%>_median_kernel_launch(&arg, flat, <%= j == "_prnan" ? 1 : 0 %>);
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
  @param [Numeric,Array,Range] axis  Finds <%=name%> along the axis.
  @param [TrueClass] keepdims  If true, the reduced axes are left in the result array as dimensions with size one.
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
    // or rather than assign: cumo_na_reduce_dimension may have set
    // CUMO_NDF_KEEP_DIM by then, and assigning would drop it
    ndf.flag |= CUMO_NDF_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP;
    v = cumo_na_ndloop(&ndf, 2, self, reduce);
    return <%=type_name%>_extract(v);
}
