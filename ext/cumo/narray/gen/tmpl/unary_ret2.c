void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    char   *p1, *p2, *p3;
    ssize_t s1, s2, s3;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR(lp, 0, p1, s1);
    CUMO_INIT_PTR(lp, 1, p2, s2);
    CUMO_INIT_PTR(lp, 2, p3, s3);

    <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,p2,p3,s1,s2,s3,n);
}

/*
  <%=name%> of self.
  @overload <%=name%>
  @return [Cumo::<%=real_class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_STRIDE_LOOP, 1,2, ain,aout};

    return cumo_na_ndloop(&ndf, 1, self);
}
