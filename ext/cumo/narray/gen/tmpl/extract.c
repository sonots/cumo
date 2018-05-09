static VALUE
<%=c_func(0)%>_cpu(VALUE self);

/*
  Returns self.
  @overload extract
  @return [Cumo::NArray]
  --- Note that Cumo::NArray always returns NArray and does not
  return a Ruby numeric object as Numo::NArray does to avoid
  synchronization between CPU and GPU for performance.

  Call `Cumo.enable_compatible_mode` to make this method behave
  compatible with Numo, or you can use `extract_cpu` method instead.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    if (cumo_compatible_mode_enabled_p()) {
        return <%=c_func(0)%>_cpu(self);
    }
    return self;
}
