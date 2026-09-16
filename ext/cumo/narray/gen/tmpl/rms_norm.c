void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *pg, char *py, uint64_t rows, uint64_t cols, double eps);

/*
  Normalizes self along its last axis by the root mean square of the row, then
  scales by gamma. Each row is taken on its own:
  y = x / sqrt(mean(x * x) + eps) * gamma, where the mean divides by the row
  length. Unlike layer_norm the row is not centred first, which is what
  Llama and the models after it normalize with.
  @overload rms_norm(gamma, eps:1e-5)
  @param [Cumo::<%=class_name%>] gamma  scale, one-dimensional and as long as the last axis.
  @param [Float] eps  added to the mean square before the square root.
  Answers a new array: inplace! is not honoured.
  @return [Cumo::<%=class_name%>] returns the normalized array, shaped like self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    VALUE x=self, gamma, eps, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("eps")};
    VALUE opts[] = {Qundef};
    VALUE x_cont, gamma_cont;
    cumo_narray_t *nx, *ngamma, *ny;
    char *x_ptr, *gamma_ptr, *y_ptr;
    size_t cols, rows;
    double double_eps = 1e-5;

    rb_scan_args(argc, argv, "1:", &gamma, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 1, opts);
    eps = cumo_option_value(opts[0], Qnil);
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }

    CUMO_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CHECK_NARRAY_TYPE(gamma, cT);

    x_cont = cumo_na_as_contiguous_array(x);
    gamma_cont = cumo_na_as_contiguous_array(gamma);

    CumoGetNArray(x_cont, nx);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "rms_norm needs an axis to normalize along");
    }
    y = cumo_na_new(cT, nx->ndim, nx->shape);

    // Taking a pointer runs allocate, which is Ruby, so all three are taken
    // before anything is measured and the checks below run over what is left.
    // See row_method.h for why the pointers themselves have to be looked at
    // again.
    x_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gamma_ptr = cumo_na_get_offset_pointer_for_read(gamma_cont);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    // cumo_na_as_contiguous_array answers what dup gave it, which a class is
    // free to define, so what these carry is not what was checked above.
    CUMO_CHECK_NARRAY_TYPE(x_cont, cT);
    CUMO_CHECK_NARRAY_TYPE(gamma_cont, cT);
    CumoGetNArray(x_cont, nx);
    CumoGetNArray(gamma_cont, ngamma);
    CumoGetNArray(y, ny);

    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "rms_norm needs an axis to normalize along");
    }
    cols = nx->shape[nx->ndim - 1];
    if (ngamma->ndim != 1 || ngamma->shape[0] != cols) {
        rb_raise(cumo_na_eShapeError, "gamma must be 1-dimensional and %"SZF"u long", cols);
    }
    CUMO_ROW_CHECK_SAME_SHAPE(ny, nx);
    CUMO_ROW_CHECK_READ_BUFFER(x_cont, x_ptr, "self");
    CUMO_ROW_CHECK_READ_BUFFER(gamma_cont, gamma_ptr, "gamma");
    CUMO_ROW_CHECK_WRITE_BUFFER(y, y_ptr, "the result");

    if (nx->size == 0) {
        return y;
    }
    rows = nx->size / cols;

    <%="cumo_#{c_iter}_kernel_launch"%>(x_ptr, gamma_ptr, y_ptr, (uint64_t)rows, (uint64_t)cols, double_eps);

    return y;
}
