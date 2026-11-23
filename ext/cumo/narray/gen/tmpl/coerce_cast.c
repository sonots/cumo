/*
  Returns nil. #store calls this on an NArray of a class it cannot store.
  It stores the result if that is of the class asked for. A class that is
  not built in can define it to convert itself.
  @overload coerce_cast(type)
    @param [Class] type  the class #store asks for.
    @return [nil]
*/
static VALUE
<%=c_func(1)%>(VALUE self, VALUE type)
{
    return Qnil;
}
