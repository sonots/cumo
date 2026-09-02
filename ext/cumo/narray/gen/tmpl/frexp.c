void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&a3,&indexer);
}

/*
  split the number x into a normalized fraction and an exponent.
  Returns [mantissa, exponent], where x = mantissa * 2**exponent.

  @overload <%=name%>(x)
  @param [Cumo::NArray,Numeric]  x
  @return [Cumo::<%=class_name%>,Cumo::Int32]  mantissa and exponent.

*/
static VALUE
<%=c_func(1)%>(VALUE mod, VALUE a1)
{
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[2] = {{cT,0},{cumo_cInt32,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 1,2, ain,aout };
    return cumo_na_ndloop(&ndf, 1, a1);
}
