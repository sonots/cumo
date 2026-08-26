#undef int_t
#define int_t uint64_t

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_reduction_arg_t* arg);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_bit_reduction_arg_t arg = cumo_na_make_bit_reduction_arg(lp, 1);
    <%="cumo_#{c_iter}_kernel_launch"%>(&arg);
}

static VALUE
<%=c_func(-1)%>_cpu(int argc, VALUE *argv, VALUE self);

/*
  Returns the number of bits.
  If argument is supplied, return Int-array counted along the axes.
  @overload <%=op_map%>(axis:nil, keepdims:false)
  @param [Integer,Array,Range] axis (keyword) axes to be counted.
  @param [TrueClass] keepdims (keyword) If true, the reduced axes are left in the result array as dimensions with size one.
  @return [Cumo::UInt64]
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE *argv, VALUE self)
{
    if (cumo_compatible_mode_enabled_p()) {
        return <%=c_func(-1)%>_cpu(argc, argv, self);
    } else {
        VALUE v, reduce;
        cumo_narray_t *na;
        cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cumo_sym_reduce,0}};
        cumo_ndfunc_arg_out_t aout[1] = {{cumo_cUInt64,0}};
        cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_FLAT_REDUCE|CUMO_NDF_INDEXER_LOOP, 2, 1, ain, aout };

        CumoGetNArray(self,na);
        if (CUMO_NA_SIZE(na)==0) {
            return INT2FIX(0);
        }
        reduce = cumo_na_reduce_dimension(argc, argv, 1, &self, &ndf, 0);
        if (cumo_na_has_idx_p(self)) {
            // The reduction addresses its input by stride, so an index array
            // has to go first. cumo_na_copy moves whole bytes and a Bit element
            // is one bit, so the copy has to be this class's own.
            VALUE copy = <%=find_tmpl("copy").c_func%>(self);
            v = cumo_na_ndloop(&ndf, 2, copy, reduce);
        } else {
            v = cumo_na_ndloop(&ndf, 2, self, reduce);
        }
        return v;
    }
}
