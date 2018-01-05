<% if type_name == 'robject' || name == 'map' %>
<% else %>
void <%="#{c_iter}_index_index_kernel_launch"%>(char *p1, char *p2, size_t *idx1, size_t *idx2, size_t N);
void <%="#{c_iter}_index_stride_kernel_launch"%>(char *p1, char *p2, size_t *idx1, ssize_t s2, size_t N);
void <%="#{c_iter}_stride_index_kernel_launch"%>(char *p1, char *p2, ssize_t s1, size_t *idx2, size_t N);
void <%="#{c_iter}_stride_stride_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, size_t N);
void <%="#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, size_t N);
<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t  n;
    char   *p1, *p2;
    ssize_t s1, s2;
    size_t *idx1, *idx2;

    INIT_COUNTER(lp, n);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    <% if type_name == 'robject' || name == 'map' %>
    {
        size_t i;
        dtype x;
        if (idx1) {
            if (idx2) {
                for (i=0; i<n; i++) {
                    GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                for (i=0; i<n; i++) {
                    GET_DATA_INDEX(p1,idx1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_STRIDE(p2,s2,dtype,x);
                }
            }
        } else {
            if (idx2) {
                for (i=0; i<n; i++) {
                    GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_INDEX(p2,idx2,dtype,x);
                }
            } else {
                //<% if need_align %>
                if (is_aligned(p1,sizeof(dtype)) &&
                    is_aligned(p2,sizeof(dtype)) ) {
                    if (s1 == sizeof(dtype) &&
                        s2 == sizeof(dtype) ) {
                        for (i=0; i<n; i++) {
                            ((dtype*)p2)[i] = m_<%=name%>(((dtype*)p1)[i]);
                        }
                        return;
                    }
                    if (is_aligned_step(s1,sizeof(dtype)) &&
                        is_aligned_step(s2,sizeof(dtype)) ) {
                        //<% end %>
                        for (i=0; i<n; i++) {
                            *(dtype*)p2 = m_<%=name%>(*(dtype*)p1);
                            p1 += s1;
                            p2 += s2;
                        }
                        return;
                        //<% if need_align %>
                    }
                }
                for (i=0; i<n; i++) {
                    GET_DATA_STRIDE(p1,s1,dtype,x);
                    x = m_<%=name%>(x);
                    SET_DATA_STRIDE(p2,s2,dtype,x);
                }
                //<% end %>
            }
        }
    }
    <% else %>
    {
        if (idx1) {
            if (idx2) {
                <%="#{c_iter}_index_index_kernel_launch"%>(p1,p2,idx1,idx2,n);
            } else {
                <%="#{c_iter}_index_stride_kernel_launch"%>(p1,p2,idx1,s2,n);
            }
        } else {
            if (idx2) {
                <%="#{c_iter}_stride_index_kernel_launch"%>(p1,p2,s1,idx2,n);
            } else {
                //<% if need_align %>
                if (is_aligned(p1,sizeof(dtype)) &&
                    is_aligned(p2,sizeof(dtype)) ) {
                    if (s1 == sizeof(dtype) &&
                        s2 == sizeof(dtype) ) {
                        <%="#{c_iter}_contiguous_kernel_launch"%>(p1,p2,n);
                        return;
                    }
                    if (is_aligned_step(s1,sizeof(dtype)) &&
                        is_aligned_step(s2,sizeof(dtype)) ) {
                        //<% end %>
                        <%="#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,n);
                        return;
                        //<% if need_align %>
                    }
                }
                <%="#{c_iter}_stride_stride_kernel_launch"%>(p1,p2,s1,s2,n);
                //<% end %>
            }
        }
    }
    <% end %>
}

/*
  Unary <%=name%>.
  @overload <%=op_map%>
  @return [Cumo::<%=class_name%>] <%=name%> of self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    ndfunc_arg_in_t ain[1] = {{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = {<%=c_iter%>, FULL_LOOP, 1,1, ain,aout};

    <% if name == 'map' %>
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    <% end %>
    return na_ndloop(&ndf, 1, self);
}
