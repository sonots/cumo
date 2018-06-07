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
  cutype =
    case type_name
    when 'sfloat'
      'float'
    when 'dfloat'
      'double'
    when 'scomplex'
      'cuComplex'
    when 'dcomplex'
      'cuDoubleComplex'
    end
%>

#define args_t <%=name%>_args_t

typedef struct {
    // enum CBLAS_ORDER order; // cuBLAS does not have order (row-major or column-major) option
    // cublasOperation_t transa, transb;
    // cublasSideMode_t side;
    // cublasFillMode_t uplo;
    // cublasDiagType_t diag;
    dtype alpha, beta;
    int m, n, k;
} args_t;

typedef struct {
    int ld;
    cublasOperation_t trans;
} gemm_layout_t;

static bool
is_f_contiguous(na_loop_args_t* arg) {
    size_t step = arg->elmsz;
    if (arg->ndim == 0) {
        return true;
    }
    if (arg->ndim == 1 && arg->shape[0] == 1) {
        return true;
    }
    for (int i = 0; i < arg->ndim; ++i) {
        if (arg->shape[i] == 1) continue;
        if (arg->iter[i].step != step) {
            return false;
        }
        step *= arg->shape[i];
    }
    return true;
}

static bool
is_c_contiguous(na_loop_args_t* arg) {
    size_t step = arg->elmsz;
    if (arg->ndim == 0) {
        return true;
    }
    if (arg->ndim == 1 && arg->shape[0] == 1) {
        return true;
    }
    for (int i = arg->ndim - 1; i >= 0 ; --i) {
        if (arg->shape[i] == 1) continue;
        if (arg->iter[i].step != step) {
            return false;
        }
        step *= arg->shape[i];
    }
    return true;
}

static gemm_layout_t
make_gemm_layout(na_loop_args_t* arg)
{
    assert(arg->ndim == 2);
    gemm_layout_t layout;
    if (is_f_contiguous(arg)) {
        layout.ld = arg->shape[0];
        layout.trans = CUBLAS_OP_T;
    } else if (is_c_contiguous(arg)) {
        layout.ld = arg->shape[1];
        layout.trans = CUBLAS_OP_N;  // transposed
    } else {
        // TODO(sonots): Make contiguous array and compute with it
        rb_raise(nary_eOperationError, "Gemm does not support non-contiguous NArray yet");
    }
    return layout;
}

extern int na_debug_flag;  // narray.c

static void
print_gemm_args(args_t* g, gemm_layout_t* a_layout, gemm_layout_t* b_layout)
{
    printf("transb=%d transa=%d, n=%d, m=%d, k=%d, ldb=%d, lda=%d, ldc=n=%d\n",
            (int)b_layout->trans,
            (int)a_layout->trans,
            (int)g->n,
            (int)g->m,
            (int)g->k,
            (int)b_layout->ld,
            (int)a_layout->ld,
            (int)g->n);
}

static void
<%=c_iter%>(na_loop_t *const lp)
{
    dtype *a, *b;
    gemm_layout_t a_layout, b_layout;
    dtype *c;
    args_t *g;
    cublasHandle_t handle = 0;
    cublasStatus_t status = 0;

    a = (dtype*)NDL_PTR(lp,0);
    b = (dtype*)NDL_PTR(lp,1);
    c = (dtype*)NDL_PTR(lp,2);
    g = (args_t*)(lp->opt_ptr);

    // TODO(sonots): Use gemmStridedBatched to support ndim >= 2 in batch

    a_layout = make_gemm_layout(&lp->args[0]);
    b_layout = make_gemm_layout(&lp->args[1]);

    // Note that cuBLAS uses the column major matrix representation.
    // We use technic which following site describes:
    // https://www.christophlassner.de/using-blas-from-c-with-row-major-data.html
    //
    // b^T = nxk matrix
    // a^T = kxm matrix
    // c^T = nxm matrix
    // c^T = b^T * a^T
    //
    // cublasSgemm(handle,transb,transa,n,m,k,&alpha,b,ldb,a,lda,&beta,c,ldc=n);

    // TODO(sonots): Cache cublas handle for each cuda device and cpu thread
    cublasCreate(&handle);
    if (na_debug_flag) {
        print_gemm_args(g, &a_layout, &b_layout);
    }
    status = cublas<%=func_prefix%>gemm(
            handle,
            b_layout.trans,
            a_layout.trans,
            g->n,
            g->m,
            g->k,
            (<%=cutype%>*)(&g->alpha),
            (<%=cutype%>*)b,
            b_layout.ld,
            (<%=cutype%>*)a,
            a_layout.ld,
            (<%=cutype%>*)(&g->beta),
            (<%=cutype%>*)c,
            g->n);
    cublasDestroy(handle);
    cumo_cuda_cublas_check_status(status);
}

/*
<%
  # ext/numo/linalg/blas/gen/decl.rb

  def mat(v,*a,**h)
    tp = h[:type] || class_name
    a.map!{|x| x==:inplace ? "inplace allowed" : x}
    a.unshift ">=2-dimentional NArray"
    "@param #{v} [#{tp}]  matrix (#{a.join(', ')})."
  end

  def opt(v,tp=nil,*a)
    tp ||= "String or Symbol"
    case v
    when "alpha"
      "@param #{v} [Float]  (default=1.0)"
    when "beta"
      "@param #{v} [Float]  (default=0.0)"
    else
      "@param #{v} [#{tp}]  #{a[0]}"
    end
  end
%>
<%
 args_v = "a, b, [c, alpha:1, beta:0]"
 params = [
   mat("a"),
   mat("b"),
   mat("c","optional",:inplace),
   opt("alpha"),
   opt("beta"),
 ].select{|x| x}.join("\n  ")
%>
  @overload <%=name%>(<%=args_v%>)
  <%=params%>
  @return [<%=class_name%>] returns c = alpha\*op( A )\*op( B ) + beta\*C.
<%=description%>
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    VALUE     a=self, b, c=Qnil, alpha, beta;
    narray_t *na1, *na2;
    size_t    out_shape[2];
    ndfunc_arg_in_t ain[3] = {{cT,2},{cT,2},{OVERWRITE,2}};
    ndfunc_arg_out_t aout[1] = {{cT,2,out_shape}};
    ndfunc_t ndf = {<%=c_iter%>, NO_LOOP, 3, 0, ain, aout};

    args_t g;
    VALUE kw_hash = Qnil;
    ID kw_table[4] = {rb_intern("alpha"),rb_intern("beta")};
    VALUE opts[4] = {Qundef,Qundef};

    rb_scan_args(argc, argv, "11:", &b, &c, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 4, opts);
    alpha    = option_value(opts[0],Qnil);
    g.alpha  = RTEST(alpha) ? m_num_to_data(alpha) : m_one;
    beta     = option_value(opts[1],Qnil);
    g.beta   = RTEST(beta)  ? m_num_to_data(beta)  : m_zero;

    GetNArray(a,na1);
    GetNArray(b,na2);
    CHECK_DIM_GE(na1,2);
    CHECK_DIM_GE(na2,2);

    g.m = ROW_SIZE(na1);
    g.k = COL_SIZE(na1);
    g.n = COL_SIZE(na2);

    if (ROW_SIZE(na2) != g.k) {
        rb_raise(nary_eShapeError,"row size of b %d must equal to col size of a %d", ROW_SIZE(na2), g.k);
    }

    if (c == Qnil) { // c is not given.
        ndfunc_arg_in_t ain_init = {sym_init,0};
        ain[2] = ain_init;
        ndf.nout = 1;
        c = INT2FIX(0);
        out_shape[0] = g.m;
        out_shape[1] = g.n;
    } else {
        narray_t *na3;
        COPY_OR_CAST_TO(c,cT);
        GetNArray(c,na3);
        CHECK_DIM_GE(na3,2);
        if (ROW_SIZE(na3) != g.m) {
            rb_raise(nary_eShapeError,"row size of c %d must equal to row size of a %d", ROW_SIZE(na3), g.m);
        }
        if (COL_SIZE(na3) != g.n) {
            rb_raise(nary_eShapeError,"col size of c %d must equal to col size of b %d", COL_SIZE(na3), g.n);
        }
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

#undef args_t
