<% unless type_name == 'robject' %>
void <%="#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, size_t n);
void <%="#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, size_t n);
void <%="#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, size_t n);
void <%="#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t n);
<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    <% if type_name == 'robject' %>
    {
        dtype x;
        if (idx1) {
            if (idx2) {
                for (; i--;) {
                    GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                for (; i--;) {
                    GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_STRIDE(p2,s2,dtype,x);
                }
            }
        } else {
            if (idx2) {
                for (; i--;) {
                    GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                for (; i--;) {
                    GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_STRIDE(p2,s2,dtype,x);
                }
            }
        }
    }
    <% else %>
    {
        if (idx1) {
            if (idx2) {
                <%="#{c_iter}_index_index_kernel_launch"%>(p1,p2,idx1,idx2,i);
            } else {
                <%="#{c_iter}_index_stride_kernel_launch"%>(p1,p2,idx1,s2,i);
            }
        } else {
            if (idx2) {
                <%="#{c_iter}_stride_index_kernel_launch"%>(p1,p2,s1,idx2,i);
            } else {
                <%="#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,i);
            }
        }
    }
    <% end %>
}

/*
  Calculate <%=name%>(x).
  @overload <%=name%>(x)
  @param [Cumo::NArray,Numeric] x  input value
  @return [Cumo::<%=class_name%>] result of <%=name%>(x).
*/
static VALUE
<%=c_func(1)%>(VALUE mod, VALUE a1)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP, 1, 1, ain, aout };

    return na_ndloop(&ndf, 1, a1);
}
