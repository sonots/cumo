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

#include "cublas_v2.h"
#include "cumo/cuda/cublas.h"

#define args_t <%=name%>_args_t

typedef struct {
  // enum CBLAS_ORDER order; // cuBLAS does not have order (row-major or column-major) option
  cublasOperation_t trans;
  cublasSideMode_t side;
  cublasFillMode_t uplo;
  cublasDiagType_t diag;
  dtype alpha, beta;
  int m, n;
} args_t;

static void
<%=c_iter%>(na_loop_t *const lp)
{
    dtype *a;
    char *p1;
    ssize_t s1;
    char *p2;
    ssize_t s2;
    int lda;
    args_t *g;

    a = (dtype*)NDL_PTR(lp,0);
    INIT_PTR(lp,1,p1,s1);
    INIT_PTR(lp,2,p2,s2);
    g = (args_t*)(lp->opt_ptr);

    lda = NDL_STEP(lp,0) / sizeof(dtype);

    // Note that cuBLAS uses the FORTRAN-order matrix representation.
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublas<%=func_prefix%>gemv(handle, g->trans, g->m, g->n, &(g->alpha), a, lda, (dtype*)p1, s1/sizeof(dtype), &(g->beta), (dtype*)p2, s2/sizeof(dtype));
    cublasDestroy(handle);
}

/*<%
  # ext/numo/linalg/blas/gen/decl.rb

  def vec(v,*a,**h)
    tp = h[:type] || class_name
    a.map!{|x| x==:inplace ? "inplace allowed" : x}
    a.unshift ">=1-dimentional NArray"
    "@param #{v} [#{tp}]  vector (#{a.join(', ')})."
  end

  def mat(v,*a,**h)
    tp = h[:type] || class_name
    a.map!{|x| x==:inplace ? "inplace allowed" : x}
    a.unshift ">=2-dimentional NArray"
    "@param #{v} [#{tp}]  matrix (#{a.join(', ')})."
  end

  def opt(v,tp=nil,*a)
    tp ||= "String or Symbol"
    case v
    when /^order$/
      "@param #{v} [#{tp}]  if 'R': Row-major, if 'C': Column-major. (default='R')"
    when /^uplo$/
      "@param #{v} [#{tp}]  if 'U': Upper triangle, if 'L': Lower triangle. (default='U')"
    when /^side$/
      "@param #{v} [#{tp}]  if 'L': op(A)\\*B (left-side op), if 'R': B\\*op(A) (right-side op). (default='L')"
    when /^diag$/
      "@param #{v} [#{tp}]  if 'U': assumed to be unit triangular, if 'N': not assumed to be unit triangular. (default='U')"
    when /^trans(\w+)?$/
      b = a[0] || $1
      "@param #{v} [#{tp}]  if 'N': Not transpose #{b}, if 'T': Transpose #{b}. (default='N')"
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
 args_v = "a, x, [y, alpha:1, beta:0, trans:'N']"
 params = [
   mat("a"),
   vec("x"),
   vec("y","optional",:inplace),
   opt("alpha"),
   opt("beta"),
   opt("trans"),
 ].select{|x| x}.join("\n  ")
%>
  @overload <%=name%>(<%=args_v%>)
  <%=params%>
  @return [<%=class_name%>] returns y = alpha*op(A)\*x + beta\*y.

<%=description%>

*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    VALUE     a=self, x, y=Qnil, alpha, beta;
    narray_t *na1, *na2;
    int   ma, na, nx;
    int   tmp;
    size_t    shape[1];
    ndfunc_arg_in_t ain[4] = {{cT,2},{cT,1},{OVERWRITE,1},{sym_init,0}};
    ndfunc_arg_out_t aout[1] = {{cT,1,shape}};
    ndfunc_t ndf = {<%=c_iter%>, NO_LOOP, 3, 0, ain, aout};

    args_t g;
    VALUE kw_hash = Qnil;
    ID kw_table[3] = {rb_intern("alpha"),rb_intern("beta"),rb_intern("trans")};
    VALUE opts[6] = {Qundef,Qundef,Qundef,Qundef,Qundef};

    rb_scan_args(argc, argv, "11:", &x, &y, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 3, opts);
    alpha   = option_value(opts[0],Qnil);
    g.alpha = RTEST(alpha) ? m_num_to_data(alpha) : m_one;
    beta    = option_value(opts[1],Qnil);
    g.beta  = RTEST(beta)  ? m_num_to_data(beta)  : m_zero;
    g.trans = option_trans(opts[2]);

    GetNArray(a,na1);
    CHECK_DIM_GE(na1,2);
    ma = ROW_SIZE(na1);
    na = COL_SIZE(na1);

    GetNArray(x,na2);
    CHECK_DIM_GE(na2,1);
    nx = COL_SIZE(na2);
    SWAP_IFTR(g.trans, ma, na, tmp);
    g.m = ma;
    g.n = na;
    CHECK_INT_EQ("na",na,"nx",nx);
    shape[0] = ma;

    if (y == Qnil) { // c is not given.
        ndf.nout = 1;
        ain[2] = ain[3];
        y = INT2FIX(0);
        shape[0] = ma;
    } else {
        narray_t *na3;
        COPY_OR_CAST_TO(y,cT);
        GetNArray(y,na3);
        CHECK_DIM_GE(na3,1);
        CHECK_SIZE_GE(na3,nx);
    }
    {
        VALUE ans;
        ans = na_ndloop3(&ndf, &g, 3, a, x, y);

        if (ndf.nout == 1) { // c is not given.
            return ans;
        } else {
            return y;
        }
    }
}

#undef args_t
