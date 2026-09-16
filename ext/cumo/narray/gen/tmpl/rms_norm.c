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
    VALUE eps, y;
    VALUE kw_hash = Qnil;
    ID kw_table[] = {rb_intern("eps")};
    VALUE opts[] = {Qundef};
    cumo_row_args_t a;
    double double_eps = 1e-5;

    rb_scan_args(argc, argv, "1:", &a.ops[0].value, &kw_hash);
    rb_get_kwargs(kw_hash, kw_table, 0, 1, opts);
    eps = cumo_option_value(opts[0], Qnil);
    if (eps != Qnil) {
        double_eps = NUM2DBL(eps);
    }
    a.ops[0].name = "gamma";

    y = cumo_row_prepare(self, cT, "rms_norm", &a, 1);
    <%="cumo_#{c_iter}_kernel_launch"%>(a.x_ptr, a.ops[0].ptr, a.y_ptr,
                                        (uint64_t)a.rows, (uint64_t)a.cols, double_eps);

    return y;
}
