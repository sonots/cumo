static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    ssize_t  i, s1, s2;
    size_t   p1;
    char    *p2;
    size_t  *idx1, *idx2;
    <%=dtype%> x;
    BIT_DIGIT *a1;
    BIT_DIGIT  y;

    // TODO(sonots): CUDA kernelize
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx2) {
        if (idx1) {
            for (; i--;) {
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1+*idx1, y); idx1++;
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1, y); p1+=s1;
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1+*idx1, y); idx1++;
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1, y); p1+=s1;
            }
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
