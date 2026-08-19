<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    size_t  i;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;
    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    {
        dtype x;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        // The index arrays are filled by a kernel; wait for it before reading
        // them from here.
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        if (idx1) {
            if (idx2) {
                for (; i--;) {
                    CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    CUMO_SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                for (; i--;) {
                    CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
                }
            }
        } else {
            if (idx2) {
                for (; i--;) {
                    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    CUMO_SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                for (; i--;) {
                    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    CUMO_SET_DATA_STRIDE(p2,s2,dtype,x);
                }
            }
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
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
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    //<% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 1, 1, ain, aout };
    <% end %>

    return cumo_na_ndloop(&ndf, 1, a1);
}
