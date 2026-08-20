<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

<%   [64,32].each do |i| %>
<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_<%=name%><%=nan%>_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg);
<% end %>

#define idx_t int<%=i%>_t
static void
<%=c_iter%>_arg<%=i%><%=nan%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t   n, idx;
        char    *d_ptr, *o_ptr;
        ssize_t  d_step;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR(lp, 0, d_ptr, d_step);
        o_ptr = CUMO_NDL_PTR(lp,1);

        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=nan%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        idx = f_<%=name[3..5]%>_index<%=nan%>(n,d_ptr,d_step);
        *(idx_t*)o_ptr = (idx_t)idx;
    }
    <% else %>
    {
        cumo_na_reduction_arg_t arg = cumo_na_make_reduction_arg(lp, 1);
        cumo_<%=type_name%>_<%=name%><%=nan%>_int<%=i%>_kernel_launch(&arg);
    }
    <% end %>
}
#undef idx_t
<% end;end %>

/*
  <%=name%>. Returns an index of the <%=name[3..5]%>imum value along the axis. See also `<%=name[3..5]%>_index`.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN posision if exist).
<% else %>
  @overload <%=name%>(axis:nil)
<% end %>
  @param [Numeric,Array,Range] axis  Finds <%=name[3..5]%>imum values along the axis and returns indices along the axis.
  @return [Integer,Cumo::Int] returns result indices.
  @example
      Cumo::NArray[3,4,1,2].<%=name%> => <% if name == 'argmin' %>2<% else %>1<% end %>
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    <% if type_name == 'robject' %>
    {
        cumo_narray_t *na;
        VALUE reduce;
        cumo_ndfunc_arg_in_t ain[2] = {{Qnil,0},{cumo_sym_reduce,0}};
        cumo_ndfunc_arg_out_t aout[1] = {{0,0,0}};
        cumo_ndfunc_t ndf = {0, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_EXTRACT, 2,1, ain,aout};

        CumoGetNArray(self,na);
        if (na->ndim==0) {
            return INT2FIX(0);
        }
        if (na->size > (~(u_int32_t)0)) {
            aout[0].type = cumo_cInt64;
            ndf.func = <%=c_iter%>_arg64;
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
        } else {
            aout[0].type = cumo_cInt32;
            ndf.func = <%=c_iter%>_arg32;
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
        }

        return cumo_na_ndloop(&ndf, 2, self, reduce);
    }
    <% else %>
    {
        cumo_narray_t *na;
        VALUE reduce, ret;
        <% if is_float %>
        cumo_na_iter_func_t iter_nan;
        <% end %>
        cumo_ndfunc_arg_in_t ain[2] = {{Qnil,0},{cumo_sym_reduce,0}};
        cumo_ndfunc_arg_out_t aout[1] = {{0,0,0}};
        cumo_ndfunc_t ndf = {0, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_EXTRACT|CUMO_NDF_INDEXER_LOOP, 2,1, ain,aout};

        CumoGetNArray(self,na);
        if (na->ndim==0) {
            return INT2FIX(0);
        }
        if (na->size > (~(u_int32_t)0)) {
            aout[0].type = cumo_cInt64;
            ndf.func = <%=c_iter%>_arg64;
            <% if is_float %>
            iter_nan = <%=c_iter%>_arg64_nan;
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        } else {
            aout[0].type = cumo_cInt32;
            ndf.func = <%=c_iter%>_arg32;
            <% if is_float %>
            iter_nan = <%=c_iter%>_arg32_nan;
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, iter_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        }

        // The index the kernel answers counts along the operand's memory, so
        // it only means the same thing as the logical index when the operand is
        // contiguous.
        if (cumo_na_check_contiguous(self) != Qtrue) {
            VALUE copy = cumo_na_copy(self);
            ret = cumo_na_ndloop(&ndf, 2, copy, reduce);
        } else {
            ret = cumo_na_ndloop(&ndf, 2, self, reduce);
        }
        if (cumo_compatible_mode_enabled_p()) {
            return rb_funcall(ret, rb_intern("extract_cpu"), 0);
        } else {
            return ret;
        }
    }
    <% end %>
}
