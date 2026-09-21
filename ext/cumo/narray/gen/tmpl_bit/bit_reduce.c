void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_pred_reduction_arg_t* arg);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_bit_pred_reduction_arg_t arg = cumo_na_make_bit_pred_reduction_arg(lp, 1);
    <%="cumo_#{c_iter}_kernel_launch"%>(&arg);
}

/*
<% case name
   when /^any/ %>
  Return true if any of bits is one (true).
<% when /^all/ %>
  Return true if all of bits are one (true).
<% end %>
  Reduced whole with no argument, answer true or false. Named axes, or a
  [:sum, true] mark on the receiver, answer the Bit they reduced to.
  @overload <%=op_map%>(axis:nil, keepdims:false)
  @param [Integer,Array,Range] axis (keyword) axes to be reduced.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [TrueClass,FalseClass,Cumo::Bit] .
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    cumo_narray_t *na;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cumo_cBit,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_EMPTY_IDENTITY, 2,1, ain,aout};

    CumoGetNArray(self,na);
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    if (CUMO_NA_SIZE(na)==0) {
        // Every bit of no bits is one, and one of no bits is not. Handing that
        // back as it is saves a launch and a wait reading it off the device.
        int keepdims = CUMO_NDF_TEST(&ndf, CUMO_NDF_KEEP_DIM) != 0;

        if (!CUMO_NDF_TEST(&ndf, CUMO_NDF_AXES_NAMED) &&
            cumo_na_reduce_ndim(self, reduce, keepdims) == 0) {
            return (INT2FIX(<%=init_bit%>) == INT2FIX(1)) ? Qtrue : Qfalse;
        }
        v = cumo_na_reduce_empty(self, reduce, cumo_cBit, keepdims);
        <%=find_tmpl("fill").c_func%>(v, INT2FIX(<%=init_bit%>));
    } else if (cumo_na_has_idx_p(self)) {
        // The reduction addresses its input by stride, so an index array has to
        // go first. cumo_na_copy moves whole bytes and a Bit element is one
        // bit, so the copy has to be this class's own.
        VALUE copy = <%=find_tmpl("copy").c_func%>(self);
        v = cumo_na_ndloop(&ndf, 2, copy, reduce);
    } else {
        v = cumo_na_ndloop(&ndf, 2, self, reduce);
    }
    // Named axes ask for an array, and so does anything the reduction left. The
    // question used to be put to the number of arguments, which counts a
    // keyword that only restates a default.
    CumoGetNArray(v,na);
    if (CUMO_NDF_TEST(&ndf, CUMO_NDF_AXES_NAMED) || na->ndim > 0) {
        return v;
    }
    // Nothing is left to index, and these three answer with a Ruby boolean
    // rather than the zero-dimensional Bit that extract returns.
    return (<%=find_tmpl("extract_cpu").c_func%>(v) == INT2FIX(1)) ? Qtrue : Qfalse;
}
