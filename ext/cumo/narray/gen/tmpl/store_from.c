//<% unless c_iter.include? 'robject' %>
void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n);
void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n);
void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n);
void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n);
//<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i, s1, s2;
    char   *p1, *p2;
    size_t *idx1, *idx2;

    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    //<% if c_iter.include? 'robject' %>
    SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    {
        <%=dtype%> x;
        dtype y;
        if (idx2) {
            if (idx1) {
                for (; i--;) {
                    GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    SET_DATA_INDEX(p1,idx1,dtype,y);
                }
            } else {
                for (; i--;) {
                    GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    SET_DATA_STRIDE(p1,s1,dtype,y);
                }
            }
        } else {
            if (idx1) {
                for (; i--;) {
                    GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    SET_DATA_INDEX(p1,idx1,dtype,y);
                }
            } else {
                for (; i--;) {
                    GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                    y = <%=macro%>(x);
                    SET_DATA_STRIDE(p1,s1,dtype,y);
                }
            }
        }
    }
    //<% else %>
    {
        if (idx2) {
            if (idx1) {
                <%="cumo_#{c_iter}_index_index_kernel_launch"%>(p1,p2,idx1,idx2,i);
            } else {
                <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(p1,p2,s1,idx2,i);
            }
        } else {
            if (idx1) {
                <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(p1,p2,idx1,s2,i);
            } else {
                <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,i);
            }
        }
    }
    //<% end %>
}


static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{Qnil,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP, 2, 0, ain, 0 };

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}
