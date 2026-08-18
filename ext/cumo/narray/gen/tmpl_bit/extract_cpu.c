/*
  Extract an element only if self is a dimensionless NArray.
  @overload extract_cpu
  @return [Numeric,Cumo::NArray]
  --- Extract element value as Ruby Object if self is a dimensionless NArray,
  otherwise returns self.
*/

static VALUE
<%=c_func(0)%>(VALUE self)
{
    CUMO_BIT_DIGIT *ptr, val;
    size_t pos;
    cumo_narray_t *na;
    CumoGetNArray(self,na);

    if (na->ndim==0) {
        pos = cumo_na_get_offset(self);
        ptr = (CUMO_BIT_DIGIT*)cumo_na_get_pointer_for_read(self);

        CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("<%=name%>", "<%=type_name%>");
        cumo_cuda_runtime_check_status(cudaDeviceSynchronize());

        {
            // Reading through the managed pointer faults the whole page back from
            // the device, which costs more than the kernel that wrote it.
            CUMO_BIT_DIGIT word;
            cumo_cuda_runtime_check_status(cudaMemcpy(&word, ptr+(pos)/CUMO_NB, sizeof(CUMO_BIT_DIGIT), cudaMemcpyDeviceToHost));
            val = (word >> ((pos)%CUMO_NB)) & 1u;
        }
        cumo_na_release_lock(self);
        return INT2FIX(val);
    }
    return self;
}
