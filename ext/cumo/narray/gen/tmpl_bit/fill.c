void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a3, CUMO_BIT_DIGIT y, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n, CUMO_BIT_DIGIT y);

static int
<%=c_iter%>_is_flat(cumo_na_bit_iarray_stridx_t* a, cumo_na_indexer_t* indexer)
{
    return indexer->ndim == 1 &&
        CUMO_SDX_IS_STRIDE(a->stridx[0]) && CUMO_SDX_GET_STRIDE(a->stridx[0]) == 1;
}

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_bit_iarray_stridx_t a3 = cumo_na_make_bit_iarray_stridx(&lp->args[0]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    CUMO_BIT_DIGIT y;
    VALUE x = lp->option;

    if (x==INT2FIX(0) || x==Qfalse) {
        y = 0;
    } else
    if (x==INT2FIX(1) || x==Qtrue) {
        y = ~(CUMO_BIT_DIGIT)0;
    } else {
        rb_raise(rb_eArgError, "invalid value for Bit");
    }

    // A word at a time is worth a separate kernel, but it needs the elements
    // laid out end to end, which after the loop is contracted means one
    // dimension of step one.
    if (<%=c_iter%>_is_flat(&a3,&indexer)) {
        <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(
            a3.ptr + a3.pos / CUMO_NB, a3.pos % CUMO_NB, indexer.total_size, y);
    } else {
        <%="cumo_#{c_iter}_kernel_launch"%>(&a3, y & 1, &indexer);
    }
}

/*
  Fill elements with other.
  @overload <%=name%> other
  @param [Numeric] other
  @return [Cumo::<%=class_name%>] self.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE val)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{cumo_sym_option}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP, 2,0, ain,0};

    cumo_na_ndloop(&ndf, 2, self, val);
    return self;
}
