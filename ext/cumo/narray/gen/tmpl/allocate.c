static VALUE
<%=c_func(0)%>(VALUE self)
{
    narray_t *na;
    void *ptr;

    GetNArray(self,na);

    switch(NA_TYPE(na)) {
    case NARRAY_DATA_T:
        ptr = NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            <% if is_object %>
            ptr = xmalloc(sizeof(dtype) * na->size);
            {   size_t i;
                VALUE *a = (VALUE*)ptr;
                for (i=na->size; i--;) {
                    *a++ = Qnil;
                }
            }
            <% else %>
            cudaError_t status = cudaMallocManaged(&ptr, sizeof(dtype) * na->size, cudaMemAttachGlobal);
            cumo_cuda_runtime_check_status(status);
            <% end %>
            NA_DATA_PTR(na) = ptr;
        }
        break;
    case NARRAY_VIEW_T:
        rb_funcall(NA_VIEW_DATA(na), rb_intern("allocate"), 0);
        break;
    case NARRAY_FILEMAP_T:
        //ptr = ((narray_filemap_t*)na)->ptr;
        // to be implemented
    default:
        rb_bug("invalid narray type : %d",NA_TYPE(na));
    }
    return self;
}
