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
            ptr = cumo_cuda_runtime_malloc(((na->size-1)/8/sizeof(CUMO_BIT_DIGIT)+1)*sizeof(CUMO_BIT_DIGIT));
            CUMO_NA_DATA_PTR(na) = ptr;
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
