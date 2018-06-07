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

typedef struct {
    dtype alpha, beta;
    int m, n, k;
} gemm_args_t;

typedef struct {
    int ld;
    cublasOperation_t trans;
} gemm_layout_t;

static bool
is_f_contiguous(VALUE a)
{
    int i;
    ssize_t s0;
    narray_t *na;

    switch(RNARRAY_TYPE(a)) {
    case NARRAY_DATA_T:
    case NARRAY_FILEMAP_T:
        return TEST_COLUMN_MAJOR(a);
    case NARRAY_VIEW_T:
        GetNArray(a, na);

        // not contiguous if it has index
        for (i = 0; i < NA_NDIM(na); ++i) {
            if (NA_IS_INDEX_AT(na, i)) return false;
        }

        // check f-contiguous
        s0 = nary_element_stride(a);
        for (i = 0; i < NA_NDIM(na); ++i) {
            if (NA_SHAPE(na)[i] == 1) continue;
            if (NA_STRIDE_AT(na, i) != s0) return false;
            s0 *= NA_SHAPE(na)[i];
        }
        return true;
    default:
        rb_raise(rb_eArgError, "NArray type : %d is not supported", RNARRAY_TYPE(a));
    }
}

static bool
is_c_contiguous(VALUE a)
{
    return na_check_contiguous(a) == Qtrue;
}

static gemm_layout_t
make_gemm_layout(VALUE a)
{
    assert(RNARRAY_NDIM(a) == 2);
    gemm_layout_t layout;
    if (is_f_contiguous(a)) {
        layout.ld = RNARRAY_SHAPE(a)[0];
        layout.trans = CUBLAS_OP_T;
    } else if (is_c_contiguous(a)) {
        layout.ld = RNARRAY_SHAPE(a)[1];
        layout.trans = CUBLAS_OP_N;  // transposed
    } else {
        // TODO(sonots): Make contiguous array and compute with it
        rb_raise(nary_eOperationError, "Gemm does not support non-contiguous NArray yet");
    }
    return layout;
}

extern int na_debug_flag;  // narray.c

static void
print_gemm_args(gemm_args_t* g, gemm_layout_t* a_layout, gemm_layout_t* b_layout)
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
<%=c_iter%>(VALUE a, VALUE b, VALUE c, gemm_args_t *g)
{
    gemm_layout_t a_layout, b_layout;
    cublasHandle_t handle = 0;
    cublasStatus_t status = 0;

    // TODO(sonots): Use gemmStridedBatched to support ndim >= 2 in batch

    a_layout = make_gemm_layout(a);
    b_layout = make_gemm_layout(b);

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
            (<%=cutype%>*)na_get_pointer_for_read(b),
            b_layout.ld,
            (<%=cutype%>*)na_get_pointer_for_read(a),
            a_layout.ld,
            (<%=cutype%>*)(&g->beta),
            (<%=cutype%>*)na_get_pointer_for_write(c),
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
    narray_t *na, *nb;
    size_t    out_shape[2];

    gemm_args_t g;
    VALUE kw_hash = Qnil;
    ID kw_table[2] = {rb_intern("alpha"), rb_intern("beta")};
    VALUE opts[2] = {Qundef, Qundef};

    rb_scan_args(argc, argv, "11:", &b, &c, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 2, opts);
    alpha   = option_value(opts[0],Qnil);
    g.alpha = RTEST(alpha) ? m_num_to_data(alpha) : m_one;
    beta    = option_value(opts[1],Qnil);
    g.beta  = RTEST(beta)  ? m_num_to_data(beta)  : m_zero;

    GetNArray(a, na);
    GetNArray(b, nb);

    // TODO(sonots): support ndim > 2
    CHECK_DIM_EQ(na, 2);
    CHECK_DIM_EQ(nb, 2);

    g.m = ROW_SIZE(na);
    g.k = COL_SIZE(na);
    g.n = COL_SIZE(nb);

    if (ROW_SIZE(nb) != g.k) {
        rb_raise(nary_eShapeError,"row size of b %d must equal to col size of a %d", (int)ROW_SIZE(nb), g.k);
    }

    if (c == Qnil) { // c is not given.
        out_shape[0] = g.m;
        out_shape[1] = g.n;
        c = nary_new(cT, 2, out_shape);
    } else {
        narray_t *nc;
        COPY_OR_CAST_TO(c, cT);
        GetNArray(c, nc);
        CHECK_DIM_EQ(nc, 2);
        if ((int)ROW_SIZE(nc) != g.m) {
            rb_raise(nary_eShapeError,"row size of c %d must equal to row size of a %d", (int)ROW_SIZE(nc), g.m);
        }
        if ((int)COL_SIZE(nc) != g.n) {
            rb_raise(nary_eShapeError,"col size of c %d must equal to col size of b %d", (int)COL_SIZE(nc), g.n);
        }
    }

    <%=c_iter%>(a, b, c, &g);
    return c;
}
