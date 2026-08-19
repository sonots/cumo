<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_bit_iarray_t* a2, cumo_na_indexer_t* indexer);
<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    <% if type_name == 'robject' %>
    {
        size_t    i, n;
        char     *p1;
        CUMO_BIT_DIGIT *a2;
        size_t    p2;
        ssize_t   s1, s2;
        size_t   *idx1;
        dtype x;
        CUMO_BIT_DIGIT b;

        CUMO_INIT_COUNTER(lp, n);
        CUMO_INIT_PTR_IDX(lp, 0, p1, s1, idx1);
        CUMO_INIT_PTR_BIT(lp, 1, a2, p2, s2);
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
        cumo_na_iarray_t a1 = cumo_na_make_iarray(&lp->args[0]);
        cumo_na_bit_iarray_t a2 = cumo_na_make_bit_iarray(&lp->args[1]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
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
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_FULL_LOOP, 1, 1, ain, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_STRIDE_LOOP|CUMO_NDF_INDEXER_LOOP, 1, 1, ain, aout };
    <% end %>

    return cumo_na_ndloop(&ndf, 1, self);
}
