<% if type_name == 'robject' %>
<% else %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, int* intmin);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    size_t  n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);
    {
        size_t i;
        dtype x;
        <%=dtype%> y;
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
        if (idx1) {
            if (idx2) {
                for (i=0; i<n; i++) {
                    CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
                    y = m_<%=name%>(x);
                    CUMO_SET_DATA_INDEX(p2,idx2,<%=dtype%>,y);
                }
            } else {
                for (i=0; i<n; i++) {
                    CUMO_GET_DATA_INDEX(p1,idx1,dtype,x);
                    y = m_<%=name%>(x);
                    CUMO_SET_DATA_STRIDE(p2,s2,<%=dtype%>,y);
                }
            }
        } else {
            if (idx2) {
                for (i=0; i<n; i++) {
                    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
                    y = m_<%=name%>(x);
                    CUMO_SET_DATA_INDEX(p2,idx2,<%=dtype%>,y);
                }
            } else {
                for (i=0; i<n; i++) {
                    CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
                    y = m_<%=name%>(x);
                    CUMO_SET_DATA_STRIDE(p2,s2,<%=dtype%>,y);
                }
            }
        }
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        //<% if is_int and !is_unsigned and name == 'abs' %>
        int *intmin = cumo_cuda_runtime_error_flag_new();
        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer,intmin);
        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (cumo_cuda_runtime_error_flag_get(intmin)) {
            lp->err_type = cumo_na_eValueError;
        }
        //<% else %>
        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer,0);
        //<% end %>
    }
    <% end %>
}


/*
  <%=name%> of self.
  @overload <%=name%>
  @return [Cumo::<%=real_class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_ndfunc_arg_in_t ain[1] = {{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{<%=result_class%>,0}};
    //<% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 1, 1, ain, aout };
    <% end %>

    return cumo_na_ndloop(&ndf, 1, self);
}
