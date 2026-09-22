/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract_cpu
  @return [Numeric,Cumo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
  This method is compatible with Numo NArray's `extract` method.
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    volatile VALUE v;
    char *ptr;
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    if (na->ndim==0) {
        // Reading through the managed pointer faults the whole page back from
        // the device, which costs more than the kernel that wrote it.
        dtype x;
        ptr = cumo_na_get_pointer_for_read(self) + cumo_na_get_offset(self);
        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cumo_cuda_runtime_memcpy_to_host(&x, ptr, sizeof(dtype)));
        v = m_extract((char*)&x);
        cumo_na_release_lock(self);
        return v;
    }
    return self;
}
