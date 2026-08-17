<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, size_t *idx1, ssize_t s2, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, CUMO_BIT_DIGIT *a2, size_t p2, ssize_t s1, ssize_t s2, uint64_t n);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t    n;
    char     *p1;
    CUMO_BIT_DIGIT *a2;
    size_t    p2;
    ssize_t   s1, s2;
    size_t   *idx1;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_BIT(lp, 1, a2, p2, s2);

    <% if type_name == 'robject' %>
    {
        size_t i;
        dtype x;
        CUMO_BIT_DIGIT b;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        if (idx1) {
            for (i=0; i<n; i++) {
                CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
                b = (m_<%=name%>(x)) ? 1:0;
                CUMO_STORE_BIT(a2,p2,b);
                p2+=s2;
            }
        } else {
            for (i=0; i<n; i++) {
                CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
                b = (m_<%=name%>(x)) ? 1:0;
                CUMO_STORE_BIT(a2,p2,b);
                p2+=s2;
            }
        }
    }
    <% else %>
    {
        if (idx1) {
            <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,a2,p2,idx1,s2,n);
        } else {
            <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,a2,p2,s1,s2,n);
        }
    }
    <% end %>
}

/*
  Condition of <%=name%>.
  @overload <%=name%>
  @return [Cumo::Bit] Condition of <%=name%>.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cumo_cBit,0}};
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 1, ain, aout };

    return cumo_na_ndloop(&ndf, 1, self);
}
