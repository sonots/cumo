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
    int stride; // in element count
    cublasOperation_t trans;
    VALUE a;
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
    narray_t *na;
    gemm_layout_t layout;

    GetNArray(a, na);
    if (is_f_contiguous(a)) {
        layout.ld = ROW_SIZE(na);
        layout.trans = CUBLAS_OP_T;
        layout.a = a;
    } else {
        layout.ld = COL_SIZE(na);
        layout.trans = CUBLAS_OP_N;  // transposed
        // force c-contiguous
        if (is_c_contiguous(a)) {
            layout.a = a;
        } else {
            if (na_debug_flag) printf("gemm: dup\n");
            layout.a = rb_funcall(a, rb_intern("dup"), 0);
        }
    }
    layout.stride = ROW_SIZE(na) * COL_SIZE(na);
    return layout;
}

extern int na_debug_flag;  // narray.c

static void
print_gemm_args(gemm_args_t* g, gemm_layout_t* a_layout, gemm_layout_t* b_layout, int stridec, int batch_count)
{
    printf("transb=%d transa=%d, n=%d, m=%d, k=%d, ldb=%d, lda=%d, ldc=n=%d, strideb=%d, stridea=%d stridec=%d batch_count=%d\n",
            (int)b_layout->trans,
            (int)a_layout->trans,
            (int)g->n,
            (int)g->m,
            (int)g->k,
            (int)b_layout->ld,
            (int)a_layout->ld,
            (int)g->n,
            (int)b_layout->stride,
            (int)a_layout->stride,
            (int)stridec,
            (int)batch_count);
}

static void
<%=c_iter%>(VALUE a, VALUE b, VALUE c, gemm_args_t *g)
{
    gemm_layout_t a_layout, b_layout;
    cublasHandle_t handle = 0;
    cublasStatus_t status = 0;
    narray_t* nc;

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

    a_layout = make_gemm_layout(a);
    b_layout = make_gemm_layout(b);

    GetNArray(c, nc);
    int stridec = ROW_SIZE(nc) * COL_SIZE(nc);
    int batch_count = NA_SIZE(nc) / stridec;

    // TODO(sonots): Cache cublas handle for each cuda device and cpu thread
    cublasCreate(&handle);
    if (na_debug_flag) print_gemm_args(g, &a_layout, &b_layout, stridec, batch_count);
    status = cublas<%=func_prefix%>gemmStridedBatched(
            handle,
            b_layout.trans,
            a_layout.trans,
            g->n,
            g->m,
            g->k,
            (<%=cutype%>*)(&g->alpha),
            (<%=cutype%>*)(na_get_pointer_for_read(b_layout.a) + na_get_offset(b_layout.a)),
            b_layout.ld,
            b_layout.stride,
            (<%=cutype%>*)(na_get_pointer_for_read(a_layout.a) + na_get_offset(a_layout.a)),
            a_layout.ld,
            a_layout.stride,
            (<%=cutype%>*)(&g->beta),
            (<%=cutype%>*)(na_get_pointer_for_write(c) + na_get_offset(c)),
            g->n,
            stridec,
            batch_count);
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
    CHECK_DIM_GE(na, 2);
    CHECK_DIM_GE(nb, 2);

    if (ROW_SIZE(nb) != COL_SIZE(na)) {
        rb_raise(nary_eShapeError,"ROW_SIZE(b)=%d must equal to COL_SIZE(a)=%d", (int)ROW_SIZE(nb), (int)COL_SIZE(na));
    }

    g.m = ROW_SIZE(na);
    g.k = COL_SIZE(na);
    g.n = COL_SIZE(nb);

    if (c == Qnil) { // c is not given.
        int ndim = NA_NDIM(na);
        if (ndim > 2) {
            size_t *shape = ALLOCA_N(size_t, ndim);
            memcpy(shape, NA_SHAPE(na), sizeof(size_t) * ndim);
            // shape[ndim - 2] = g.m;
            shape[ndim - 1] = g.n;
            c = nary_new(cT, ndim, shape);
        } else {
            size_t shape[2];
            shape[0] = g.m;
            shape[1] = g.n;
            c = nary_new(cT, 2, shape);
        }
    } else {
        narray_t *nc;
        COPY_OR_CAST_TO(c, cT);
        GetNArray(c, nc);
        CHECK_DIM_GE(nc, 2);
        if (ROW_SIZE(nc) != ROW_SIZE(na)) {
            rb_raise(nary_eShapeError,"ROW_SIZE(c)=%d must equal to ROW_SIZE(a)=%d", (int)ROW_SIZE(nc), (int)ROW_SIZE(na));
        }
        if (COL_SIZE(nc) != COL_SIZE(nb)) {
            rb_raise(nary_eShapeError,"COL_SIZE(c)=%d must equal to COL_SIZE(b)=%d", (int)COL_SIZE(nc), (int)COL_SIZE(nc));
        }
    }

    <%=c_iter%>(a, b, c, &g);
    return c;
}
