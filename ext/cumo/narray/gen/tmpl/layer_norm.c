void <%="cumo_#{c_iter}_kernel_launch"%>(char *px, char *pg, char *pb, char *py, uint64_t rows, uint64_t cols, double eps);

/*
  Normalizes self along its last axis, then scales by gamma and shifts by beta.
  Each row is taken on its own: y = (x - mean) / sqrt(var + eps) * gamma + beta,
  where mean and var are of that row and var divides by the row length.
  @overload layer_norm(gamma, beta, eps:1e-5)
  @param [Cumo::<%=class_name%>] gamma  scale, one-dimensional and as long as the last axis.
  @param [Cumo::<%=class_name%>] beta  shift, one-dimensional and as long as the last axis.
  @param [Float] eps  added to the variance before the square root.
  Answers a new array: inplace! is not honoured.
  @return [Cumo::<%=class_name%>] returns the normalized array, shaped like self.
*/
static VALUE
<%=c_func(-1)%>(int argc, VALUE argv[], VALUE self)
{
    VALUE eps, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("eps")};
    VALUE opts[] = {Qundef};
    cumo_row_args_t a;
    double double_eps = 1e-5;

    rb_scan_args(argc, argv, "2:", &a.ops[0].value, &a.ops[1].value, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 1, opts);
    eps = cumo_option_value(opts[0], Qnil);
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    a.ops[0].name = "gamma";
    a.ops[1].name = "beta";

    y = cumo_row_prepare(self, cT, "layer_norm", &a, 2);
    <%="cumo_#{c_iter}_kernel_launch"%>(a.x_ptr, a.ops[0].ptr, a.ops[1].ptr, a.y_ptr,
                                        (uint64_t)a.rows, (uint64_t)a.cols, double_eps);

    return y;
}
