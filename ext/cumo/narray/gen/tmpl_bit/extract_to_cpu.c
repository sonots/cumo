/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract_to_cpu
  @return [Numeric,Cumo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
*/

static VALUE
<%=c_func(0)%>(VALUE self)
{
    BIT_DIGIT *ptr, val;
    size_t pos;
    narray_t *na;
    GetNArray(self,na);

    if (na->ndim==0) {
        pos = na_get_offset(self);
        ptr = (BIT_DIGIT*)na_get_pointer_for_read(self);

        SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

        val = ((*((ptr)+(pos)/NB)) >> ((pos)%NB)) & 1u;
        na_release_lock(self);
        return INT2FIX(val);
    }
    return self;
}
