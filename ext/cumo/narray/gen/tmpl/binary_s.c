<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer);
void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int scalar_is_left);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t  i;
        char    *p1, *p2, *p3;
        ssize_t s1, s2, s3;
        dtype x, y;

        CUMO_INIT_COUNTER(lp, i);
        CUMO_INIT_PTR(lp, 0, p1, s1);
        CUMO_INIT_PTR(lp, 1, p2, s2);
        CUMO_INIT_PTR(lp, 2, p3, s3);
        CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        for (; i--;) {
            CUMO_GET_DATA_STRIDE(p1,s1,dtype,x);
            CUMO_GET_DATA_STRIDE(p2,s2,dtype,y);
            x = m_<%=name%>(x,y);
            CUMO_SET_DATA_STRIDE(p3,s3,dtype,x);
        }
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

<% unless type_name == 'robject' %>
// A Ruby numeric operand rides in through opt_ptr instead of being cast to a
// 0-dimensional array, which costs a kernel launch to fill. Either side of a
// module function can be the numeric one, so there is an iterator for each.
static void
<%=c_iter%>_s(cumo_na_loop_t *const lp)
{
    dtype sv = *(dtype*)(lp->opt_ptr);
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_s_kernel_launch"%>(&a1,&a3,&indexer,sv,0);
}

static void
<%=c_iter%>_s_left(cumo_na_loop_t *const lp)
{
    dtype sv = *(dtype*)(lp->opt_ptr);
    cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t a3 = cumo_na_make_iarray(&lp->args[1]);
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

    <%="cumo_#{c_iter}_s_kernel_launch"%>(&a1,&a3,&indexer,sv,1);
}
<% end %>

/*
  Calculate <%=name%>(a1,a2).
  @overload <%=name%>(a1,a2)
  @param [Cumo::NArray,Numeric] a1  first value
  @param [Cumo::NArray,Numeric] a2  second value
  @return [Cumo::<%=class_name%>] <%=name%>(a1,a2).
*/
static VALUE
<%=c_func(2)%>(VALUE mod, VALUE a1, VALUE a2)
{
    cumo_ndfunc_arg_in_t ain[2] = {{cT,0},{cT,0}};
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP, 2, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 2, 1, ain, aout };

    {
        cumo_ndfunc_arg_in_t ain_s[1] = {{cT,0}};
        int n1 = RTEST(rb_obj_is_kind_of(a1, rb_cNumeric));
        int n2 = RTEST(rb_obj_is_kind_of(a2, rb_cNumeric));

        if (!n1 && n2) {
            dtype sv = m_num_to_data(a2);
            cumo_ndfunc_t ndf_s = { <%=c_iter%>_s, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 1, 1, ain_s, aout };
            return cumo_na_ndloop3(&ndf_s, &sv, 1, a1);
        }
        if (n1 && !n2) {
            dtype sv = m_num_to_data(a1);
            cumo_ndfunc_t ndf_s = { <%=c_iter%>_s_left, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 1, 1, ain_s, aout };
            return cumo_na_ndloop3(&ndf_s, &sv, 1, a2);
        }
    }
    <% end %>

    return cumo_na_ndloop(&ndf, 2, a1, a2);
}

