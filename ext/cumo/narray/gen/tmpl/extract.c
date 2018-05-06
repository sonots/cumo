/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract
  @return [Numeric,Cumo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    return self;
}
