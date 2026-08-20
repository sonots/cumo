<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_minmax_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2);
<% if is_float %>
void cumo_<%=type_name%>_minmax_nan_kernel_launch(cumo_na_reduction_arg_t* arg, cumo_na_iarray_t* out2);
<% end %>
<% end %>

<% (is_float ? ["","_nan"] : [""]).each do |j| %>
static void
<%=c_iter%><%=j%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t   n;
        char    *p1;
        ssize_t  s1;
        dtype    xmin,xmax;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, p1, s1);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=j%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        f_<%=name%><%=j%>(n,p1,s1,&xmin,&xmax);

        *(dtype*)(lp->args[1].ptr + lp->args[1].iter[0].pos) = xmin;
        *(dtype*)(lp->args[2].ptr + lp->args[2].iter[0].pos) = xmax;
    }
    <% else %>
    {
        cumo_na_reduction_arg_t arg = cumo_na_make_reduction_arg(lp, 1);
        cumo_na_iarray_t out2 = cumo_na_make_iarray_given_ndim(&lp->args[2], arg.out_indexer.ndim);
        cumo_<%=type_name%>_minmax<%=j%>_kernel_launch(&arg, &out2);
    }
    <% end %>
}
<% end %>

/*
  <%=name%> of self.
<% if is_float %>
  @overload <%=name%>(axis:nil, keepdims:false, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN if exist).
<% else %>
  @overload <%=name%>(axis:nil, keepdims:false)
<% end %>
  @param [Numeric,Array,Range] axis  Finds min-max along the axis.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::<%=class_name%>,Cumo::<%=class_name%>] min and max of self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE reduce, ret;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_EXTRACT, 2,2, ain,aout};

  <% if is_float %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_nan);
  <% else %>
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
  <% end %>
    //<% unless type_name == 'robject' %>
    ndf.flag |= CUMO_NDF_INDEXER_LOOP;
    //<% end %>
    if (cumo_na_has_idx_p(self)) {
        VALUE copy = cumo_na_copy(self); // reduction does not support idx, make contiguous
        ret = cumo_na_ndloop(&ndf, 2, copy, reduce);
    } else {
        ret = cumo_na_ndloop(&ndf, 2, self, reduce);
    }
    // ndloop ignores CUMO_NDF_EXTRACT, so each method carrying it extracts for itself.
    if (cumo_compatible_mode_enabled_p()) {
        return rb_assoc_new(rb_funcall(RARRAY_AREF(ret,0), rb_intern("extract_cpu"), 0),
                            rb_funcall(RARRAY_AREF(ret,1), rb_intern("extract_cpu"), 0));
    }
    return ret;
}
