//<% if ['sfloat', 'dfloat', 'scomplex', 'dcomplex'].include?(type_name) %>
<%
  func_prefix =
    case type_name
    when 'sfloat'
      'S'
    when 'dfloat'
      'D'
    when 'scomplex'
      'C'
    when 'dcomplex'
      'Z'
    end
%>

// TODO(sonots): Move to suitable place
#include "cublas_v2.h"

#define args_t <%=name%>_args_t

typedef struct {
  // enum CBLAS_ORDER order; // cuBLAS does not have order (row-major or column-major) option
  cublasOperation_t transa, transb;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasDiagType_t diag;
  dtype alpha, beta;
  int m, n, k;
} args_t;

//#define func_p <%=func_name%>_p
//
//static <%=func_name%>_t func_p = 0;

static void
<%=c_iter%>(na_loop_t *const lp)
{
    dtype *a, *b;
    int    lda, ldb;
    dtype *c;
    int    ldc;
    args_t *g;

    a = (dtype*)NDL_PTR(lp,0);
    b = (dtype*)NDL_PTR(lp,1);
    c = (dtype*)NDL_PTR(lp,2);
    g = (args_t*)(lp->opt_ptr);

    lda = NDL_STEP(lp,0) / sizeof(dtype);
    ldb = NDL_STEP(lp,1) / sizeof(dtype);
    ldc = NDL_STEP(lp,2) / sizeof(dtype);

    //printf("m=%d n=%d k=%d\n",g->m,g->n,g->k);

    //cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&al,a,m,b,k,&bet,c,m);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublas<%=func_prefix%>gemm(handle, g->transa, g->transb, g->m, g->n, g->k, &(g->alpha), a, lda, b, ldb, &(g->beta), c, ldc);
    cublasDestroy(handle);
}

/*
<%
 args_v = "a, b, [c, alpha:1, beta:0, transa:'N', transb:'N']"
 params = [
   mat("a"),
   mat("b"),
   mat("c","optional",:inplace),
   opt("alpha"),
   opt("beta"),
   opt("transa"),
   opt("transb"),
 ].select{|x| x}.join("\n  ")
%>
  @overload <%=name%>(<%=args_v%>)
  <%=params%>
  @return [<%=class_name%>] returns c = alpha\*op( A )\*op( B ) + beta\*C.
<%=description%>
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE const argv[], VALUE UNUSED(mod))
{
    VALUE     a, b, c=Qnil, alpha, beta;
    narray_t *na1, *na2;
    int   ma, ka, kb, nb, tmp;
    size_t    shape[2];
    ndfunc_arg_in_t ain[3] = {{cT,2},{cT,2},{OVERWRITE,2}};
    ndfunc_arg_out_t aout[1] = {{cT,2,shape}};
    ndfunc_t ndf = {<%=c_iter%>, NO_LOOP, 3, 0, ain, aout};

    args_t g;
    VALUE kw_hash = Qnil;
    ID kw_table[4] = {id_alpha,id_beta,id_transa,id_transb};
    VALUE opts[6] = {Qundef,Qundef,Qundef,Qundef,Qundef,Qundef};

    CHECK_FUNC(func_p,"<%=func_name%>");

    rb_scan_args(argc, argv, "21:", &a, &b, &c, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 5+TR*2, opts);
    alpha    = option_value(opts[0],Qnil);
    g.alpha  = RTEST(alpha) ? m_num_to_data(alpha) : m_one;
    beta     = option_value(opts[1],Qnil);
    g.beta   = RTEST(beta)  ? m_num_to_data(beta)  : m_zero;
    //g.order  = option_order(opts[2]);
    g.transa = option_trans(opts[2]);
    g.transb = option_trans(opts[3]);

    GetNArray(a,na1);
    GetNArray(b,na2);
    CHECK_DIM_GE(na1,2);
    CHECK_DIM_GE(na2,2);
    ma = ROW_SIZE(na1); // m
    ka = COL_SIZE(na1); // k (lda)
    kb = ROW_SIZE(na2); // k
    nb = COL_SIZE(na2); // n (ldb)

    SWAP_IFTR(g.transa, ma, ka, tmp);
    SWAP_IFTR(g.transb, kb, nb, tmp);
    CHECK_INT_EQ("ka",ka,"kb",kb);
    g.m = ma;
    g.n = nb;
    g.k = ka;

    SWAP(ma, mb, tmp);
    //SWAP_IFROW(g.order, ma,nb, tmp);

    if (c == Qnil) { // c is not given.
        ndfunc_arg_in_t ain_init = {sym_init,0};
        ain[2] = ain_init;
        ndf.nout = 1;
        c = INT2FIX(0);
        shape[0] = nb;
        shape[1] = ma;
    } else {
        narray_t *na3;
        int nc;
        COPY_OR_CAST_TO(c,cT);
        GetNArray(c,na3);
        CHECK_DIM_GE(na3,2);
        nc = ROW_SIZE(na3);
        if (nc < nb) {
            rb_raise(nary_eShapeError,"nc=%d must be >= nb=%d",nc,nb);
        }
        //CHECK_LEADING_GE("ldc",g.ldc,"m",ma);
    }
    {
        VALUE ans = na_ndloop3(&ndf, &g, 3, a, b, c);

        if (ndf.nout == 1) { // c is not given.
            return ans;
        } else {
            return c;
        }
    }
}

//#undef func_p
#undef args_t
<% end %>
