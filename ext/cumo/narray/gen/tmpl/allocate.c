<% if !is_object && !is_bit %>
void cumo_<%=type_name%>_new_dim0_kernel_launch(dtype *ptr, dtype x);
<% end %>

static VALUE
<%=c_func(0)%>(VALUE self)
{
    cumo_narray_t *na;
    void *ptr;

    CumoGetNArray(self,na);

    switch(CUMO_NA_TYPE(na)) {
    case CUMO_NARRAY_DATA_T:
        ptr = CUMO_NA_DATA_PTR(na);
        if (na->size > 0 && ptr == NULL) {
            if (na->size > SIZE_MAX / sizeof(dtype)) {
                rb_raise(rb_eRangeError, "total byte size of data is too large");
            }
            <% if is_object %>
            ptr = xmalloc(sizeof(dtype) * na->size);
            {   size_t i;
                VALUE *a = (VALUE*)ptr;
                for (i=na->size; i--;) {
                    *a++ = Qnil;
                }
            }
            <% else %>
            ptr = cumo_cuda_runtime_malloc(sizeof(dtype) * na->size);
            <% end %>
            CUMO_NA_DATA_PTR(na) = ptr;
            CUMO_NA_DATA_OWNED(na) = TRUE;
            <% unless is_object %>
            // Device memory never passes through ruby_xmalloc, so the GC sees a
            // few-byte object holding an arbitrarily large buffer and does not
            // run until the host itself runs out.
            rb_gc_adjust_memory_usage(sizeof(dtype) * na->size);
            <% end %>
        }
        <% if !is_object && !is_bit %>
        // A scalar cast lazily kept its value on the Ruby side; giving it
        // memory is where it reaches the device.
        {
            VALUE num = rb_attr_get(self, cumo_id_pending_scalar);
            if (!NIL_P(num)) {
                rb_ivar_set(self, cumo_id_pending_scalar, Qnil);
                cumo_<%=type_name%>_new_dim0_kernel_launch((dtype*)CUMO_NA_DATA_PTR(na), m_num_to_data(num));
                // Callers that read on the host synchronize before they ask for
                // the pointer, which is before this fill was even launched.
                CUMO_SHOW_SYNCHRONIZE_WARNING_ONCE("allocate", "<%=type_name%>");
                cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
            }
        }
        <% end %>
        break;
    case CUMO_NARRAY_VIEW_T:
        rb_funcall(CUMO_NA_VIEW_DATA(na), rb_intern("allocate"), 0);
        break;
    case CUMO_NARRAY_FILEMAP_T:
        //ptr = ((cumo_narray_filemap_t*)na)->ptr;
        // to be implemented
    default:
        rb_bug("invalid narray type : %d",CUMO_NA_TYPE(na));
    }
    return self;
}
