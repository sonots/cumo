void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *pg, char *pb, char *py, uint64_t rows, uint64_t cols, double eps);

/*
  Normalizes self along its last axis, then scales by gamma and shifts by beta.
  Each row is taken on its own: y = (x - mean) / sqrt(var + eps) * gamma + beta,
  where mean and var are of that row and var divides by the row length.
  @overload layer_norm(gamma, beta, eps:1e-5)
  @param [Cumo::<%=class_name%>] gamma  scale, one-dimensional and as long as the last axis.
  @param [Cumo::<%=class_name%>] beta  shift, one-dimensional and as long as the last axis.
  @param [Float] eps  added to the variance before the square root.
  @return [Cumo::<%=class_name%>] returns the normalized array, shaped like self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    VALUE x=self, gamma, beta, eps, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("eps")};
    VALUE opts[] = {Qundef};
    VALUE x_cont, gamma_cont, beta_cont;
    cumo_narray_t *nx, *ngamma, *nbeta, *ny;
    char *x_ptr, *gamma_ptr, *beta_ptr, *y_ptr;
    size_t cols, rows;
    double double_eps = 1e-5;

    rb_scan_args(argc, argv, "2:", &gamma, &beta, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 1, opts);
    eps = cumo_option_value(opts[0], Qnil);
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }

    CUMO_CHECK_NARRAY_TYPE(x, cT);
    CUMO_CHECK_NARRAY_TYPE(gamma, cT);
    CUMO_CHECK_NARRAY_TYPE(beta, cT);

    x_cont = cumo_na_as_contiguous_array(x);
    gamma_cont = cumo_na_as_contiguous_array(gamma);
    beta_cont = cumo_na_as_contiguous_array(beta);

    CumoGetNArray(x_cont, nx);
    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "layer_norm needs an axis to normalize along");
    }
    y = cumo_na_new(cT, nx->ndim, nx->shape);

    // Taking a pointer runs allocate, which is Ruby and free to resize any of
    // these or hand back a buffer of its own, so all four are taken before
    // anything is measured and nothing runs between the last of them and the
    // checks below.
    x_ptr = cumo_na_get_offset_pointer_for_read(x_cont);
    gamma_ptr = cumo_na_get_offset_pointer_for_read(gamma_cont);
    beta_ptr = cumo_na_get_offset_pointer_for_read(beta_cont);
    y_ptr = cumo_na_get_offset_pointer_for_write(y);

    // cumo_na_as_contiguous_array answers what dup gave it, which a class is
    // free to define, so what these carry is not what was checked above.
    CUMO_CHECK_NARRAY_TYPE(x_cont, cT);
    CUMO_CHECK_NARRAY_TYPE(gamma_cont, cT);
    CUMO_CHECK_NARRAY_TYPE(beta_cont, cT);
    CumoGetNArray(x_cont, nx);
    CumoGetNArray(gamma_cont, ngamma);
    CumoGetNArray(beta_cont, nbeta);
    CumoGetNArray(y, ny);

    if (nx->ndim < 1) {
        rb_raise(cumo_na_eShapeError, "layer_norm needs an axis to normalize along");
    }
    cols = nx->shape[nx->ndim - 1];
    if (ngamma->ndim != 1 || ngamma->shape[0] != cols) {
        rb_raise(cumo_na_eShapeError, "gamma must be 1-dimensional and %"SZF"u long", cols);
    }
    if (nbeta->ndim != 1 || nbeta->shape[0] != cols) {
        rb_raise(cumo_na_eShapeError, "beta must be 1-dimensional and %"SZF"u long", cols);
    }
    CUMO_CHECK_SIZE_EQ(ny->size, nx->size);

    if (nx->size == 0) {
        return y;
    }
    rows = nx->size / cols;

    <%="cumo_#{c_iter}_kernel_launch"%>(x_ptr, gamma_ptr, beta_ptr, y_ptr, (uint64_t)rows, (uint64_t)cols, double_eps);

    return y;
}
