static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    dtype   x;
    <%=dtype%> y;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    if (idx1) {
        if (idx2) {
            for (; i--;) {
                CUMO_GET_DATA(p1+*idx1,dtype,x);
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,y);
                x = m_<%=name%>(x,y);
                CUMO_SET_DATA_INDEX(p1,idx1,dtype,x);
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA(p1+*idx1,dtype,x);
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,y);
                x = m_<%=name%>(x,y);
                CUMO_SET_DATA_INDEX(p1,idx1,dtype,x);
            }
        }
    } else {
        if (idx2) {
            for (; i--;) {
                CUMO_GET_DATA(p1,dtype,x);
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,y);
                x = m_<%=name%>(x,y);
                CUMO_SET_DATA_STRIDE(p1,s1,dtype,x);
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA(p1,dtype,x);
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,y);
                x = m_<%=name%>(x,y);
                CUMO_SET_DATA_STRIDE(p1,s1,dtype,x);
            }
        }
    }
}

static VALUE
<%=c_func(1)%>(VALUE self, VALUE a1)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{<%=result_class%>,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 2, 0, ain, 0 };

    cumo_na_ndloop(&ndf, 2, self, a1);
    return a1;
}
