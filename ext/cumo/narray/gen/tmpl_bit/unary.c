static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    size_t  p1, p3;
    ssize_t s1, s3;
    size_t *idx1, *idx3;
    int     o1, l1, r1, len;
    CUMO_BIT_DIGIT *a1, *a3;
    CUMO_BIT_DIGIT  x;
    CUMO_BIT_DIGIT  y;

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a3, p3, s3, idx3);
    if (s1!=1 || s3!=1 || idx1 || idx3) {
        for (; n--;) {
            CUMO_LOAD_BIT_STEP(a1, p1, s1, idx1, x);
            y = m_<%=name%>(x);
            CUMO_STORE_BIT_STEP(a3, p3, s3, idx3, y);
        }
    } else {
        o1 =  p1-p3;
        l1 =  CUMO_NB+o1;
        r1 =  CUMO_NB-o1;
        if (p3>0 || n<CUMO_NB) {
            len = CUMO_NB - p3;
            if ((int)n<len) len=n;
            if (o1>=0) x = *a1>>o1;
            else       x = *a1<<-o1;
            if (p1+len>CUMO_NB)  x |= *(a1+1)<<r1;
            a1++;
            y = m_<%=name%>(x);
            *a3 = (y & (CUMO_SLB(len)<<p3)) | (*a3 & ~(CUMO_SLB(len)<<p3));
            a3++;
            n -= len;
        }
        if (o1==0) {
            for (; n>=CUMO_NB; n-=CUMO_NB) {
                x = *(a1++);
                y = m_<%=name%>(x);
                *(a3++) = y;
            }
        } else {
            for (; n>=CUMO_NB; n-=CUMO_NB) {
                if (o1==0) {
                    x = *a1;
                } else if (o1>0) {
                    x = *a1>>o1  | *(a1+1)<<r1;
                } else {
                    x = *a1<<-o1 | *(a1-1)>>l1;
                }
                a1++;
                y = m_<%=name%>(x);
                *(a3++) = y;
            }
        }
        if (n>0) {
            if (o1==0) {
                x = *a1;
            } else if (o1>0) {
                x = *a1>>o1;
                if ((int)n>r1) {
                    x |= *(a1+1)<<r1;
                }
            } else {
                x = *(a1-1)>>l1;
                if ((int)n>-o1) {
                    x |= *a1<<-o1;
                }
            }
            y = m_<%=name%>(x);
            *a3 = (y & CUMO_SLB(n)) | (*a3 & CUMO_BALL<<n);
        }
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
