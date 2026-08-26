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
  If argument is supplied, return Bit-array reduced along the axes.
  @overload <%=op_map%>(axis:nil, keepdims:false)
  @param [Integer,Array,Range] axis (keyword) axes to be reduced.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::Bit] .
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    VALUE v, reduce;
    cumo_narray_t *na;
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cumo_cBit,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_INDEXER_LOOP, 2,1, ain,aout};

    CumoGetNArray(self,na);
    if (CUMO_NA_SIZE(na)==0) {
        return Qfalse;
    }
    reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
    if (cumo_na_has_idx_p(self)) {
        // The reduction addresses its input by stride, so an index array has to
        // go first. cumo_na_copy moves whole bytes and a Bit element is one
        // bit, so the copy has to be this class's own.
        VALUE copy = <%=find_tmpl("copy").c_func%>(self);
        v = cumo_na_ndloop(&ndf, 2, copy, reduce);
    } else {
        v = cumo_na_ndloop(&ndf, 2, self, reduce);
    }
    if (argc > 0) {
        return v;
    }
    v = <%=find_tmpl("extract").c_func%>(v);
    switch (v) {
    case INT2FIX(0):
        return Qfalse;
    case INT2FIX(1):
        return Qtrue;
    default:
        rb_bug("unexpected result");
        return v;
    }
}
