<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_index_kernel_launch"%>(char *ptr, size_t *idx, dtype val, uint64_t n);
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *ptr, ssize_t step, dtype val, uint64_t n);
<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t   i;
    char    *p1;
    ssize_t  s1;
    size_t  *idx1;
    VALUE    x = lp->option;
    dtype    y;
    INIT_COUNTER(lp, i);
    INIT_PTR_IDX(lp, 0, p1, s1, idx1);
    y = m_num_to_data(x);
    <% if type_name == 'robject' %>
    SHOW_CPU_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    if (idx1) {
        for (; i--;) {
            SET_DATA_INDEX(p1,idx1,dtype,y);
        }
    } else {
        for (; i--;) {
            SET_DATA_STRIDE(p1,s1,dtype,y);
        }
    }
    <% else %>
    if (idx1) {
        <%="cumo_#{c_iter}_index_kernel_launch"%>(p1,idx1,y,i);
    } else {
        <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,s1,y,i);
    }
    <% end %>
}

/*
  Fill elements with other.
  @overload <%=name%> other
  @param [Numeric] other
  @return [Cumo::<%=class_name%>] self.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE val)
{
    ndfunc_arg_in_t ain[2] = {{OVERWRITE,0},{sym_option}};
    ndfunc_t ndf = { <%=c_iter%>, FULL_LOOP, 2, 0, ain, 0 };

    na_ndloop(&ndf, 2, self, val);
    return self;
}
