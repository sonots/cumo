void <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, ssize_t s1, size_t *idx1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s2, size_t *idx2, CUMO_BIT_DIGIT *a3, size_t p3, ssize_t s3, size_t *idx3, uint64_t n);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a2, size_t p2, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n);

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    size_t  p1, p2, p3;
    ssize_t s1, s2, s3;
    size_t *idx1, *idx2, *idx3;
    CUMO_BIT_DIGIT *a1, *a2, *a3;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a2, p2, s2, idx2);
    CUMO_INIT_PTR_BIT_IDX(lp, 2, a3, p3, s3, idx3);
    if (s1!=1 || s2!=1 || s3!=1 || idx1 || idx2 || idx3) {
        <%="cumo_#{c_iter}_elementwise_kernel_launch"%>(a1,p1,s1,idx1,a2,p2,s2,idx2,a3,p3,s3,idx3,n);
    } else {
        <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(a1,p1,a2,p2,a3,p3,n);
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
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 2, 1, ain, aout };

    return cumo_na_ndloop(&ndf, 2, self, other);
}
