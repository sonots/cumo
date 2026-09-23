void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_stridx_kernel_launch"%>(cumo_na_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    if (cumo_na_loop_has_index(lp)) {
        cumo_na_iarray_stridx_t b1 = cumo_na_make_iarray_stridx(&lp->args[0]);
        cumo_na_iarray_stridx_t b2 = cumo_na_make_iarray_stridx(&lp->args[1]);

        <%="cumo_#{c_iter}_stridx_kernel_launch"%>(&b1,&b2,&indexer);
    } else {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
    }
}

static VALUE
<%=c_func(1)%>(VALUE self, VALUE a1)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{<%=result_class%>,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 2, 0, ain, 0 };

    cumo_na_ndloop(&ndf, 2, self, a1);
    return a1;
}
