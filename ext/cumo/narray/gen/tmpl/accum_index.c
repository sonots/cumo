<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

<%   [64,32].each do |i| %>
<% unless type_name == 'robject' %>
void cumo_<%=type_name%>_<%=name%><%=nan%>_int<%=i%>_kernel_launch(cumo_na_reduction_arg_t* arg);
<% end %>

#define idx_t int<%=i%>_t
static void
<%=c_iter%>_index<%=i%><%=nan%>(cumo_na_loop_t *const lp)
{
    // TODO(sonots): Support nan in CUDA
    <% if type_name == 'robject' || nan == '_nan' %>
    {
        size_t   n, idx;
        char    *d_ptr, *i_ptr, *o_ptr;
        ssize_t  d_step, i_step;

        INIT_COUNTER(lp, n);
        INIT_PTR(lp, 0, d_ptr, d_step);
        INIT_PTR(lp, 1, i_ptr, i_step);
        o_ptr = NDL_PTR(lp,2);

        SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%><%=nan%>", "<%=type_name%>");
        idx = f_<%=name%><%=nan%>(n,d_ptr,d_step);
        *(idx_t*)o_ptr = *(idx_t*)(i_ptr + i_step * idx);
    }
    <% else %>
    {
        cumo_na_reduction_arg_t arg = cumo_na_make_reduction_arg(lp);
        cumo_<%=type_name%>_<%=name%><%=nan%>_int<%=i%>_kernel_launch(&arg);
    }
    <% end %>
}
#undef idx_t
<% end;end %>

/*
  <%=name%>. Return an index of result.
<% if is_float %>
  @overload <%=name%>(axis:nil, nan:false)
  @param [TrueClass] nan  If true, apply NaN-aware algorithm (return NaN posision if exist).
<% else %>
  @overload <%=name%>(axis:nil)
<% end %>
  @param [Numeric,Array,Range] axis  Affected dimensions.
  @return [Integer,Cumo::Int] returns result index of <%=name%>.
  @example
      Cumo::NArray[3,4,1,2].min_index => 3
 */
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    //<% if type_name == 'robject' %>
    {
        cumo_narray_t *na;
        VALUE idx, reduce;
        cumo_ndfunc_arg_in_t ain[3] = {{Qnil,0},{Qnil,0},{cumo_sym_reduce,0}};
        cumo_ndfunc_arg_out_t aout[1] = {{0,0,0}};
        cumo_ndfunc_t ndf = {0, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_EXTRACT, 3,1, ain,aout};

        GetNArray(self,na);
        if (na->ndim==0) {
            return INT2FIX(0);
        }
        if (na->size > (~(u_int32_t)0)) {
            aout[0].type = cumo_cInt64;
            idx = cumo_na_new(cumo_cInt64, na->ndim, na->shape);
            ndf.func = <%=c_iter%>_index64;
            <% if is_float %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_index64_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        } else {
            aout[0].type = cumo_cInt32;
            idx = cumo_na_new(cumo_cInt32, na->ndim, na->shape);
            ndf.func = <%=c_iter%>_index32;
            <% if is_float %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_index32_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        }
        rb_funcall(idx, rb_intern("seq"), 0);

        return cumo_na_ndloop(&ndf, 3, self, idx, reduce);
    }
    <% else %>
    {
        cumo_narray_t *na;
        VALUE reduce;
        cumo_ndfunc_arg_in_t ain[2] = {{Qnil,0},{cumo_sym_reduce,0}};
        cumo_ndfunc_arg_out_t aout[1] = {{0,0,0}};
        cumo_ndfunc_t ndf = {0, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_EXTRACT|CUMO_NDF_INDEXER_LOOP, 2,1, ain,aout};

        GetNArray(self,na);
        if (na->size > (~(u_int32_t)0)) {
            aout[0].type = cumo_cInt64;
            ndf.func = <%=c_iter%>_index64;
            <% if is_float %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_index64_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        } else {
            aout[0].type = cumo_cInt32;
            ndf.func = <%=c_iter%>_index32;
            <% if is_float %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, <%=c_iter%>_index32_nan);
            <% else %>
            reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
            <% end %>
        }

        return cumo_na_ndloop(&ndf, 2, self, reduce);
    }
    <% end %>
}
