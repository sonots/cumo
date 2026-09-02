void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
}

static VALUE
<%=c_func(1)%>(VALUE self, VALUE a1)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{<%=result_class%>,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 0, ain, 0 };

    cumo_na_ndloop(&ndf, 2, self, a1);
    return a1;
}
