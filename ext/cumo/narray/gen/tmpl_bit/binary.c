void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_bit_iarray_stridx_t* a2, cumo_na_bit_iarray_stridx_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a2, size_t p2, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n);

static int
<%=c_iter%>_is_flat(cumo_na_bit_iarray_stridx_t* a, cumo_na_indexer_t* indexer)
{
    return indexer->ndim == 1 &&
        CUMO_SDX_IS_STRIDE(a->stridx[0]) && CUMO_SDX_GET_STRIDE(a->stridx[0]) == 1;
}

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_bit_iarray_stridx_t a1 = cumo_na_make_bit_iarray_stridx(&lp->args[0]);
    cumo_na_bit_iarray_stridx_t a2 = cumo_na_make_bit_iarray_stridx(&lp->args[1]);
    cumo_na_bit_iarray_stridx_t a3 = cumo_na_make_bit_iarray_stridx(&lp->args[2]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    // A word at a time is worth a separate kernel, but it needs every operand
    // laid out end to end, which after the loop is contracted means one
    // dimension of step one.
    if (<%=c_iter%>_is_flat(&a1,&indexer) && <%=c_iter%>_is_flat(&a2,&indexer) && <%=c_iter%>_is_flat(&a3,&indexer)) {
        <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(
            a1.ptr + a1.pos / CUMO_NB, a1.pos % CUMO_NB,
            a2.ptr + a2.pos / CUMO_NB, a2.pos % CUMO_NB,
            a3.ptr + a3.pos / CUMO_NB, a3.pos % CUMO_NB,
            indexer.total_size);
    } else {
        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&a3,&indexer);
    }
}

/*
  Binary <%=name%>.
  @overload <%=op_map%> other
  @param [Cumo::NArray,Numeric] other
  @return [Cumo::NArray] <%=name%> of self and other.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE other)
{
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 1, ain, aout };

    return cumo_na_ndloop(&ndf, 2, self, other);
}
