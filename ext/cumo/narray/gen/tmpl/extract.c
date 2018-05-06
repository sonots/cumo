/*
  Returns self.
  @overload extract
  @return [Cumo::NArray]
  --- This method always returns self unlike Numo/NArray to avoid synchronization between GPU and CPU.
  Use "extract_to_cpu" instead to get a Ruby numeric object for 0-dimensional NArray as Numo/NArray's extract.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    return self;
}
