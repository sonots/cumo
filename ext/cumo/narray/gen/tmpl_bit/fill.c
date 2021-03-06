static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    size_t  p3;
    ssize_t s3;
    size_t *idx3;
    int     len;
    CUMO_BIT_DIGIT *a3;
    CUMO_BIT_DIGIT  y;
    VALUE x = lp->option;

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    if (x==INT2FIX(0) || x==Qfalse) {
        y = 0;
    } else
    if (x==INT2FIX(1) || x==Qtrue) {
        y = ~(CUMO_BIT_DIGIT)0;
    } else {
        rb_raise(rb_eArgError, "invalid value for Bit");
    }

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a3, p3, s3, idx3);
    if (idx3) {
        y = y&1;
        for (; n--;) {
            CUMO_STORE_BIT(a3, p3+*idx3, y); idx3++;
        }
    } else if (s3!=1) {
        y = y&1;
        for (; n--;) {
            CUMO_STORE_BIT(a3, p3, y); p3+=s3;
        }
    } else {
        if (p3>0 || n<CUMO_NB) {
            len = CUMO_NB - p3;
            if ((int)n<len) len=n;
            *a3 = (y & (CUMO_SLB(len)<<p3)) | (*a3 & ~(CUMO_SLB(len)<<p3));
            a3++;
            n -= len;
        }
        for (; n>=CUMO_NB; n-=CUMO_NB) {
            *(a3++) = y;
        }
        if (n>0) {
            *a3 = (y & CUMO_SLB(n)) | (*a3 & CUMO_BALL<<n);
        }
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
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2,0, ain,0};

    cumo_na_ndloop(&ndf, 2, self, val);
    return self;
}
