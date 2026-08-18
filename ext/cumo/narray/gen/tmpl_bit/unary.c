void <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a3, size_t p3, ssize_t s3, size_t *idx3, uint64_t n);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    size_t  p1, p3;
    ssize_t s1, s3;
    size_t *idx1, *idx3;
    CUMO_BIT_DIGIT *a1, *a3;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a3, p3, s3, idx3);
    if (s1!=1 || s3!=1 || idx1 || idx3) {
        <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(a1,p1,s1,idx1,a3,p3,s3,idx3,n);
    } else {
        <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(a1,p1,a3,p3,n);
    }
}

/*
  Unary <%=name%>.
  @overload <%=name%>
  @return [Cumo::<%=class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 1,1, ain,aout};

    return cumo_na_ndloop(&ndf, 1, self);
}
