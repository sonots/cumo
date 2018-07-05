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

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a3, p3, s3, idx3);
    CUMO_INIT_PTR_BIT_IDX(lp, 1, a1, p1, s1, idx1);
    if (s1!=1 || s3!=1 || idx1 || idx3) {
        for (; n--;) {
            CUMO_LOAD_BIT_STEP(a1, p1, s1, idx1, x);
            CUMO_STORE_BIT_STEP(a3, p3, s3, idx3, x);
        }
    } else {
        o1 =  p1 % CUMO_NB;
        o1 -= p3;
        l1 =  CUMO_NB+o1;
        r1 =  CUMO_NB-o1;
        if (p3>0 || n<CUMO_NB) {
            len = CUMO_NB - p3;
            if ((int)n<len) len=n;
            if (o1>=0) x = *a1>>o1;
            else       x = *a1<<-o1;
            if (p1+len>CUMO_NB)  x |= *(a1+1)<<r1;
            a1++;
            *a3 = (x & (CUMO_SLB(len)<<p3)) | (*a3 & ~(CUMO_SLB(len)<<p3));
            a3++;
            n -= len;
        }
        if (o1==0) {
            for (; n>=CUMO_NB; n-=CUMO_NB) {
                x = *(a1++);
                *(a3++) = x;
            }
        } else {
            for (; n>=CUMO_NB; n-=CUMO_NB) {
                x = *a1>>o1;
                if (o1<0)  x |= *(a1-1)>>l1;
                if (o1>0)  x |= *(a1+1)<<r1;
                a1++;
                *(a3++) = x;
            }
        }
        if (n>0) {
            x = *a1>>o1;
            if (o1<0)  x |= *(a1-1)>>l1;
            *a3 = (x & CUMO_SLB(n)) | (*a3 & CUMO_BALL<<n);
        }
    }
}

static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{Qnil,0}};
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2,0, ain,0};

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}
