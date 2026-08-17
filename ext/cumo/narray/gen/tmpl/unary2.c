<% if type_name == 'robject' %>
<% else %>
void <%="cumo_#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, uint64_t n, int* intmin);
void <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, uint64_t n, int* intmin);
void <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, uint64_t n, int* intmin);
void <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n, int* intmin);
void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, uint64_t n, int* intmin);

static void
<%=c_iter%>_launch(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t *idx1, size_t *idx2, size_t n, int* intmin)
{
    if (idx1) {
        if (idx2) {
            <%="cumo_#{c_iter}_index_index_kernel_launch"%>(p1,p2,idx1,idx2,n,intmin);
        } else {
            <%="cumo_#{c_iter}_index_stride_kernel_launch"%>(p1,p2,idx1,s2,n,intmin);
        }
    } else {
        if (idx2) {
            <%="cumo_#{c_iter}_stride_index_kernel_launch"%>(p1,p2,s1,idx2,n,intmin);
        } else {
            //<% if need_align %>
            if (cumo_is_aligned(p1,sizeof(dtype)) &&
                cumo_is_aligned(p2,sizeof(<%=dtype%>)) ) {
                if (s1 == sizeof(dtype) &&
                    s2 == sizeof(<%=dtype%>) ) {
                    <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(p1,p2,n,intmin);
                    return;
                }
                if (cumo_is_aligned_step(s1,sizeof(dtype)) &&
                    cumo_is_aligned_step(s2,sizeof(<%=dtype%>)) ) {
                    //<% end %>
                    <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,n,intmin);
                    return;
                    //<% if need_align %>
                }
            }
            <%="cumo_#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,n,intmin);
            //<% end %>
        }
    }
}
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;

    CUMO_INIT_COUNTER(lp, n);
    CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    <% if type_name == 'robject' %>
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
        //<% if is_int and !is_unsigned and name == 'abs' %>
        int *intmin = cumo_cuda_runtime_error_flag_new();
        <%=c_iter%>_launch(p1,p2,s1,s2,idx1,idx2,n,intmin);
        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        if (cumo_cuda_runtime_error_flag_get(intmin)) {
            lp->err_type = cumo_na_eValueError;
        }
        //<% else %>
        <%=c_iter%>_launch(p1,p2,s1,s2,idx1,idx2,n,0);
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
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 1, ain, aout };

    return cumo_na_ndloop(&ndf, 1, self);
}
