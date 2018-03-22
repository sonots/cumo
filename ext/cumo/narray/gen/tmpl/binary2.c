<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, char *p4, ssize_t s1, ssize_t s2, ssize_t s3, ssize_t s4, uint64_t n);
<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t   i, n;
    char    *p1, *p2, *p3, *p4;
    ssize_t  s1, s2, s3, s4;
    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);
    INIT_PTR(lp, 3, p4, s4);
    for (i=n; i--;) {
        <% if type_name == 'robject' %>
        {
            dtype    x, y, a, b;
            SHOW_CPU_WARNING_ONCE("<%=name%>", "<%=type_name%>");
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
<% if is_int and %w[divmod].include? name %>
                if (y==0) {
                    lp->err_type = rb_eZeroDivError;
                    return;
                }
<% end %>
            m_<%=name%>(x,y,a,b);
            SET_DATA_STRIDE(p3,s3,dtype,a);
            SET_DATA_STRIDE(p4,s4,dtype,b);
        }
        <% else %>
        <%="cumo_#{c_iter}_stride_kernel_launch"%>(p1,p2,p3,p4,s1,s2,s3,s4,n);
        <% end %>
    }
}

static VALUE
<%=c_func%>_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[2] = {{cT,0},{cT,0}};
    ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP, 2, 2, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary <%=name%>.
  @overload <%=op_map%> other
  @param [Cumo::NArray,Numeric] other
  @return [Cumo::NArray] <%=name%> of self and other.
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE other)
{
    <% if is_object %>
    return <%=c_func%>_self(self, other);
    <% else %>
    VALUE klass, v;
    klass = na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return <%=c_func%>_self(self, other);
    } else {
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, <%=id_op%>, 1, other);
    }
    <% end %>
}
