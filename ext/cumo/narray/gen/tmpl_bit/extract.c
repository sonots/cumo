static VALUE
<%=c_func(0)%>_cpu(VALUE self);

/*
  Returns self.
  @overload extract
  @return [Cumo::Bit]
  --- Note that Cumo::Bit always returns Cumo::Bit and does not
  return a Ruby Integer as Numo::Bit does to avoid
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
