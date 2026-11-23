/*
  Allocates the memory of an array made by new, which has none until then.
  The elements are not initialized. An empty array gets no memory. On a view
  it allocates the array the view looks into. An array that has its memory
  already keeps it.

  @return [Cumo::<%=class_name%>] self
*/
static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_narray_t *na;
    char *ptr;

    CumoGetNArray(self,na);

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        ptr = CUMO_NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            ptr = cumo_cuda_runtime_malloc(CUMO_BIT_BYTES(na->size));
            CUMO_NA_DATA_PTR(na) = ptr;
            rb_gc_adjust_memory_usage(CUMO_BIT_BYTES(na->size));
            cumo_na_keep_young_if_large(self, CUMO_BIT_BYTES(na->size));
        }
        break;
    case CUMO_NARRAY_VIEW_T:
        rb_funcall(CUMO_NA_VIEW_DATA(na), rb_intern("allocate"), 0);
        break;
    default:
        rb_raise(rb_eRuntimeError,"invalid narray type");
    }
    return self;
}
