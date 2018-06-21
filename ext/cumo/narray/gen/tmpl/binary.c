// TODO(sonots): handle zero division error in CUDA kernel?
// ref. https://devtalk.nvidia.com/default/topic/415951/divide-by-zero-handling/

//<% if is_int and %w[div mod divmod].include? name %>
#define check_intdivzero(y)              \
    if ((y)==0) {                        \
        lp->err_type = rb_eZeroDivError; \
        return;                          \
    }
//<% else %>
#define check_intdivzero(y) {}
//<% end %>

<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t   i, n;
        char    *p1, *p2, *p3;
        ssize_t  s1, s2, s3;

        INIT_COUNTER(lp, n);
        INIT_PTR(lp, 0, p1, s1);
        INIT_PTR(lp, 1, p2, s2);
        INIT_PTR(lp, 2, p3, s3);

        SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        //<% if need_align %>
        if (is_aligned(p1,sizeof(dtype)) &&
            is_aligned(p2,sizeof(dtype)) &&
            is_aligned(p3,sizeof(dtype)) ) {

            if (s1 == sizeof(dtype) &&
                s2 == sizeof(dtype) &&
                s3 == sizeof(dtype) ) {

                if (p1 == p3) { // inplace case
                    for (i=0; i<n; i++) {
                        check_intdivzero(*(dtype*)p2);
                        ((dtype*)p1)[i] = m_<%=name%>(((dtype*)p1)[i],((dtype*)p2)[i]);
                    }
                } else {
                    for (i=0; i<n; i++) {
                        check_intdivzero(*(dtype*)p2);
                        ((dtype*)p3)[i] = m_<%=name%>(((dtype*)p1)[i],((dtype*)p2)[i]);
                    }
                }
                return;
            }
            if (is_aligned_step(s1,sizeof(dtype)) &&
                is_aligned_step(s2,sizeof(dtype)) &&
                is_aligned_step(s3,sizeof(dtype)) ) {
                //<% end %>

                if (s2 == 0){ // Broadcasting from scalar value.
                    check_intdivzero(*(dtype*)p2);
                    if (s1 == sizeof(dtype) &&
                        s3 == sizeof(dtype) ) {
                        if (p1 == p3) { // inplace case
                            for (i=0; i<n; i++) {
                                ((dtype*)p1)[i] = m_<%=name%>(((dtype*)p1)[i],*(dtype*)p2);
                            }
                        } else {
                            for (i=0; i<n; i++) {
                                ((dtype*)p3)[i] = m_<%=name%>(((dtype*)p1)[i],*(dtype*)p2);
                            }
                        }
                    } else {
                        for (i=0; i<n; i++) {
                            *(dtype*)p3 = m_<%=name%>(*(dtype*)p1,*(dtype*)p2);
                            p1 += s1;
                            p3 += s3;
                        }
                    }
                } else { // Broadcasting from Numo::NArray
                    if (p1 == p3) { // inplace case
                        for (i=0; i<n; i++) {
                            check_intdivzero(*(dtype*)p2);
                            *(dtype*)p1 = m_<%=name%>(*(dtype*)p1,*(dtype*)p2);
                            p1 += s1;
                            p2 += s2;
                        }
                    } else {
                        for (i=0; i<n; i++) {
                            check_intdivzero(*(dtype*)p2);
                            *(dtype*)p3 = m_<%=name%>(*(dtype*)p1,*(dtype*)p2);
                            p1 += s1;
                            p2 += s2;
                            p3 += s3;
                        }
                    }
                }

                return;
                //<% if need_align %>
            }
        }
        for (i=0; i<n; i++) {
            dtype x, y, z;
            GET_DATA_STRIDE(p1,s1,dtype,x);
            GET_DATA_STRIDE(p2,s2,dtype,y);
            check_intdivzero(y);
            z = m_<%=name%>(x,y);
            SET_DATA_STRIDE(p3,s3,dtype,z);
        }
        //<% end %>
    }
    <% else %>
    {
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_iarray_t a2 = cumo_na_make_iarray(&lp->args[1]);
        cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[2]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&a3,&indexer);
    }
    <% end %>
}
#undef check_intdivzero

static VALUE
<%=c_func%>_self(VALUE self, VALUE other)
{
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP, 2, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, STRIDE_LOOP|NDF_INDEXER_LOOP, 2, 1, ain, aout };
    <% end %>

    return cumo_na_ndloop(&ndf, 2, self, other);
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

    klass = cumo_na_upcast(CLASS_OF(self),CLASS_OF(other));
    if (klass==cT) {
        return <%=c_func%>_self(self, other);
    } else {
        v = rb_funcall(klass, cumo_id_cast, 1, self);
        return rb_funcall(v, <%=cumo_id_op%>, 1, other);
    }
    <% end %>
}
