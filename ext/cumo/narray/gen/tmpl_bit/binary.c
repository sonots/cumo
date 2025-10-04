static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    size_t  p1, p2, p3;
    ssize_t s1, s2, s3;
    size_t *idx1, *idx2, *idx3;
    int     o1, o2, l1, l2, r1, r2, len;
    CUMO_BIT_DIGIT *a1, *a2, *a3;
    CUMO_BIT_DIGIT  x, y;

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a2, p2, s2, idx2);
    CUMO_INIT_PTR_BIT_IDX(lp, 2, a3, p3, s3, idx3);
    if (s1!=1 || s2!=1 || s3!=1 || idx1 || idx2 || idx3) {
        for (; n--;) {
            CUMO_LOAD_BIT_STEP(a1, p1, s1, idx1, x);
            CUMO_LOAD_BIT_STEP(a2, p2, s2, idx2, y);
            x = m_<%=name%>(x,y);
            CUMO_STORE_BIT_STEP(a3, p3, s3, idx3, x);
        }
    } else {
        o1 =  p1-p3;
        o2 =  p2-p3;
        l1 =  CUMO_NB+o1;
        r1 =  CUMO_NB-o1;
        l2 =  CUMO_NB+o2;
        r2 =  CUMO_NB-o2;
        if (p3>0 || n<CUMO_NB) {
            len = CUMO_NB - p3;
            if ((int)n<len) len=n;
            if (o1>=0) x = *a1>>o1;
            else       x = *a1<<-o1;
            if (p1+len>CUMO_NB)  x |= *(a1+1)<<r1;
            a1++;
            if (o2>=0) y = *a2>>o2;
            else       y = *a2<<-o2;
            if (p2+len>CUMO_NB)  y |= *(a2+1)<<r2;
            a2++;
            x = m_<%=name%>(x,y);
            *a3 = (x & (CUMO_SLB(len)<<p3)) | (*a3 & ~(CUMO_SLB(len)<<p3));
            a3++;
            n -= len;
        }
        if (o1==0 && o2==0) {
            for (; n>=CUMO_NB; n-=CUMO_NB) {
                x = *(a1++);
                y = *(a2++);
                x = m_<%=name%>(x,y);
                *(a3++) = x;
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
                if (o2==0) {
                    y = *a2;
                } else if (o2>0) {
                    y = *a2>>o2  | *(a2+1)<<r2;
                } else {
                    y = *a2<<-o2 | *(a2-1)>>l2;
                }
                a2++;
                x = m_<%=name%>(x,y);
                *(a3++) = x;
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
            if (o2==0) {
                y = *a2;
            } else if (o2>0) {
                y = *a2>>o2;
                if ((int)n>r2) {
                    y |= *(a2+1)<<r2;
                }
            } else {
                y = *(a2-1)>>l2;
                if ((int)n>-o2) {
                    y |= *a2<<-o2;
                }
            }
            x = m_<%=name%>(x,y);
            *a3 = (x & CUMO_SLB(n)) | (*a3 & CUMO_BALL<<n);
        }
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
