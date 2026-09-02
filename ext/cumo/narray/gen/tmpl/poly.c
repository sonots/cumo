<% unless type_name == 'robject' %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* x, cumo_na_iarray_t* coef, int ncoef, cumo_na_iarray_t* z, cumo_na_indexer_t* indexer);

// Horner over every element in one launch. The host loop this replaces ran
// once per element, each time behind a cudaDeviceSynchronize.
static void
<%=c_iter%>_kernel(cumo_na_loop_t *const lp)
{
    int i, ncoef = lp->narg - 2;
    cumo_na_indexer_t indexer = cumo_na_make_indexer(&lp->args[0]);
    cumo_na_iarray_t x = cumo_na_make_iarray(&lp->args[0]);
    cumo_na_iarray_t z = cumo_na_make_iarray(&lp->args[lp->narg - 1]);
    cumo_na_iarray_t *coef = ALLOCA_N(cumo_na_iarray_t, ncoef > 0 ? ncoef : 1);

    for (i = 0; i < ncoef; ++i) {
        coef[i] = cumo_na_make_iarray(&lp->args[i + 1]);
    }
    <%="cumo_#{c_iter}_kernel_launch"%>(&x, coef, ncoef, &z, &indexer);
}
<% end %>

<% if type_name == 'robject' %>
static void
<%=c_iter%>(cumo_na_loop_t *const lp)
{
    size_t  i, n;
    dtype  x, y, a;

    CUMO_SHOW_SYNCHRONIZE_FIXME_WARNING_ONCE("<%=name%>", "<%=type_name%>");
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    n = lp->narg - 2;
    x = *(dtype*)CUMO_NDL_PTR(lp,0);
    y = *(dtype*)CUMO_NDL_PTR(lp,n);
    for (i=1; i<n; i++) {
        y = m_mul(x,y);
        a = *(dtype*)CUMO_NDL_PTR(lp,n-i);
        y = m_add(y,a);
    }
    *(dtype*)CUMO_NDL_PTR(lp,lp->narg-1) = y;
}
<% end %>

/*
  Calculate polynomial.
    `x.poly(a0,a1,a2,...,an) = a0 + a1*x + a2*x**2 + ... + an*x**n`
  @overload <%=name%> a0, a1, ..., an
  @param [Cumo::NArray,Numeric] a0,a1,...,an
  @return [Cumo::<%=class_name%>]
*/
static VALUE
<%=c_func(-2)%>(VALUE self, VALUE args)
{
    int argc, i;
    VALUE *argv;
    volatile VALUE v, a;
    cumo_ndfunc_arg_out_t aout[1] = {{cT,0}};
    <% if type_name == 'robject' %>
    cumo_ndfunc_t ndf = { <%=c_iter%>, CUMO_NO_LOOP, 0, 1, 0, aout };
    <% else %>
    cumo_ndfunc_t ndf = { <%=c_iter%>_kernel, CUMO_STRIDE_LOOP_NIP|CUMO_NDF_INDEXER_LOOP|CUMO_NDF_ANY_ORDER, 0, 1, 0, aout };
    <% end %>

    argc = RARRAY_LEN(args);
    ndf.nin = argc+1;
    ndf.ain = ALLOCA_N(cumo_ndfunc_arg_in_t,argc+1);
    for (i=0; i<argc+1; i++) {
        ndf.ain[i].type = cT;
        ndf.ain[i].dim = 0;
    }
    argv = ALLOCA_N(VALUE,argc+1);
    argv[0] = self;
    for (i=0; i<argc; i++) {
        argv[i+1] = RARRAY_PTR(args)[i];
    }
    a = rb_ary_new4(argc+1, argv);
    v = cumo_na_ndloop2(&ndf, a);
    return <%=type_name%>_extract(v);
}
