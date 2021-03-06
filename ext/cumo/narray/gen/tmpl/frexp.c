static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t   i;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;
    dtype    x;
    int      y;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    CUMO_INIT_PTR(lp, 2, p3, s3);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    for (; i--;) {
        CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
        x = m_<%=name%>(x,&y);
        CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
        CUMO_SET_DATA_STRIDE(p3,s3,int32_t,y);
    }
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
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP, 1,2, ain,aout };
    return cumo_na_ndloop(&ndf, 1, a1);
}
