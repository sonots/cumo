// TODO(sonots): handle zero division error in CUDA kernel?
// ref. https://devtalk.nvidia.com/default/topic/415951/divide-by-zero-handling/

<% if c_iter.include?('robject') %>
<% if is_int and %w[div mod divmod].include? name %>
#define check_intdivzero(y)              \
    if ((y)==0) {                        \
        lp->err_type = rb_eZeroDivError; \
        return;                          \
    }
<% else %>
#define check_intdivzero(y) {}
<% end %>
<% end %>

<% unless c_iter.include?('robject') %>
void <%="#{c_iter}_contiguous_kernel_launch"%>(char *p1, char *p2, char *p3, size_t n);
void <%="#{c_iter}_stride_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, size_t n);
<% end %>

static void
<%=c_iter%>(na_loop_t *const lp)
{
    size_t   n;
    char    *p1, *p2, *p3;
    ssize_t  s1, s2, s3;

    INIT_COUNTER(lp, n);
    INIT_PTR(lp, 0, p1, s1);
    INIT_PTR(lp, 1, p2, s2);
    INIT_PTR(lp, 2, p3, s3);

    //<% if need_align %>
    if (is_aligned(p1,sizeof(dtype)) &&
        is_aligned(p2,sizeof(dtype)) &&
        is_aligned(p3,sizeof(dtype)) ) {

        if (s1 == sizeof(dtype) &&
            s2 == sizeof(dtype) &&
            s3 == sizeof(dtype) ) {

            // TODO(sonots): CPU warning
            <% if c_iter.include?('robject') %>
            {
                size_t i;
                for (i=0; i<n; i++) {
                    check_intdivzero(*(dtype*)p2);
                    ((dtype*)p3)[i] = m_<%=name%>(((dtype*)p1)[i],((dtype*)p2)[i]);
                }
            }
            <% else %>
            <%="#{c_iter}_contiguous_kernel_launch"%>(p1,p2,p3,n);
            <% end %>
            return;
        }
        if (is_aligned_step(s1,sizeof(dtype)) &&
            is_aligned_step(s2,sizeof(dtype)) &&
            is_aligned_step(s3,sizeof(dtype)) ) {
            //<% end %>
            <% if c_iter.include?('robject') %>
            {
                size_t i;
                for (i=0; i<n; i++) {
                    check_intdivzero(*(dtype*)p2);
                    *(dtype*)p3 = m_<%=name%>(*(dtype*)p1,*(dtype*)p2);
                    p1 += s1;
                    p2 += s2;
                    p3 += s3;
                }
            }
            <% else %>
            <%="#{c_iter}_stride_kernel_launch"%>(p1,p2,p3,s1,s2,s3,n);
            <% end %>
            return;
            //<% if need_align %>
        }
    }
    <% if c_iter.include?('robject') %>
    {
        size_t i;
        for (i=0; i<n; i++) {
            dtype x, y, z;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            check_intdivzero(y);
            z = m_<%=name%>(x,y);
            SET_DATA_STRIDE(p3,s3,dtype,z);
        }
    }
    <% else %>
    <%="#{c_iter}_stride_kernel_launch"%>(p1,p2,p3,s1,s2,s3,n);
    <% end %>
    //<% end %>
}
#undef check_intdivzero

static VALUE
<%=c_func%>_self(VALUE self, VALUE other)
{
    ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    ndfunc_arg_out_t aout[1] = {{cT,0}};
    ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP, 2, 1, ain, aout };

    return na_ndloop(&ndf, 2, self, other);
}

/*
  Binary <%=name%>.
  @overload <%=op_map%> other
  @param [Cumo::NArray,Numeric] other
  @return [Cumo::NArray] self <%=op_map%> other
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
        // TODO(sonots): CPU warning
        v = rb_funcall(klass, id_cast, 1, self);
        return rb_funcall(v, <%=id_op%>, 1, other);
    }
    <% end %>
}
