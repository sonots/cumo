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
            // Never fewer bytes than this array has held: a view made under a
            // larger shape reads through this pointer.
            size_t bytes = ((na->size-1)/8/sizeof(CUMO_BIT_DIGIT)+1)*sizeof(CUMO_BIT_DIGIT);
            if (bytes < CUMO_NA_DATA_CAPACITY(na)) {
                bytes = CUMO_NA_DATA_CAPACITY(na);
            }
            ptr = cumo_cuda_runtime_malloc(bytes);
            CUMO_NA_DATA_PTR(na) = ptr;
            CUMO_NA_DATA_OWNED(na) = TRUE;
            CUMO_NA_DATA_CAPACITY(na) = bytes;
            rb_gc_adjust_memory_usage(bytes);
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
