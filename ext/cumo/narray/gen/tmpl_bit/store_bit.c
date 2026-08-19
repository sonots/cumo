void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_bit_iarray_stridx_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n);

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
    cumo_na_bit_iarray_stridx_t a1 = cumo_na_make_bit_iarray_stridx(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    // A word at a time is worth a separate kernel, but it needs both operands
    // laid out end to end, which after the loop is contracted means one
    // dimension of step one.
    if (<%=c_iter%>_is_flat(&a1,&indexer) && <%=c_iter%>_is_flat(&a3,&indexer)) {
        <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(
            a1.ptr + a1.pos / CUMO_NB, a1.pos % CUMO_NB,
            a3.ptr + a3.pos / CUMO_NB, a3.pos % CUMO_NB,
            indexer.total_size);
    } else {
        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a3,&indexer);
    }
}

static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{Qnil,0}};
    // ndloop cannot stage a Bit operand into a buffer — a buffer copy moves
    // whole bytes — so INDEX_LOOP stays on and the kernel walks both operands'
    // own indices.
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP, 2,0, ain,0};

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}
