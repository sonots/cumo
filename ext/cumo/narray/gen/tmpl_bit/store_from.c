//<% unless c_iter.include? 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer);
//<% end %>

static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    //<% if c_iter.include? 'robject' %>
    ssize_t  i, s1, s2;
    size_t   p1;
    char    *p2;
    size_t  *idx1, *idx2;
    <%=dtype%> x;
    CUMO_BIT_DIGIT *a1;
    CUMO_BIT_DIGIT  y;

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("store_<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

    CUMO_INIT_COUNTER(lp, i);
    CUMO_INIT_PTR_BIT_IDX(lp, 0, a1, p1, s1, idx1);
    CUMO_INIT_PTR_IDX(lp, 1, p2, s2, idx2);

    if (idx2) {
        if (idx1) {
            for (; i--;) {
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1+*idx1, y); idx1++;
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA_INDEX(p2,idx2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1, y); p1+=s1;
            }
        }
    } else {
        if (idx1) {
            for (; i--;) {
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1+*idx1, y); idx1++;
            }
        } else {
            for (; i--;) {
                CUMO_GET_DATA_STRIDE(p2,s2,<%=dtype%>,x);
                y = <%=macro%>(x);
                CUMO_STORE_BIT(a1, p1, y); p1+=s1;
            }
        }
    }
    //<% else %>
    {
        cumo_na_bit_iarray_stridx_t a1 = cumo_na_make_bit_iarray_stridx(&lp->args[0]);
        cumo_na_iarray_stridx_t a2 = cumo_na_make_iarray_stridx(&lp->args[1]);
        cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);

        <%="cumo_#{c_iter}_kernel_launch"%>(&a1,&a2,&indexer);
    }
    //<% end %>
}


static VALUE
<%=c_func(:nodef)%>(VALUE self, VALUE obj)
{
    cumo_ndfunc_arg_in_t ain[2] = {{CUMO_OVERWRITE,0},{Qnil,0}};
    //<% if c_iter.include? 'robject' %>
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP, 2,0, ain,0};
    <% else %>
    // The Bit side cannot be staged into a buffer by ndloop — a buffer copy
    // moves whole bytes — so INDEX_LOOP stays on and the kernel walks both
    // operands' own indices.
    cumo_ndfunc_t ndf = {<%=c_iter%>, CUMO_FULL_LOOP|CUMO_NDF_INDEXER_LOOP, 2,0, ain,0};
    <% end %>

    cumo_na_ndloop(&ndf, 2, self, obj);
    return self;
}
