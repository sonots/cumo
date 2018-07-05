static size_t
<%=type_name%>_memsize(const void* ptr)
{
    size_t size = sizeof(cumo_narray_data_t);
    const cumo_narray_data_t *na = (const cumo_narray_data_t*)ptr;

    assert(na->base.type == CUMO_NARRAY_DATA_T);

    if (na->ptr != NULL) {
  <% if is_bit %>
        size += ((na->base.size-1)/8/sizeof(BIT_DIGIT)+1)*sizeof(BIT_DIGIT);
  <% else %>
        size += na->base.size * sizeof(dtype);
  <% end %>
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            size += sizeof(size_t) * na->base.ndim;
        }
    }
    return size;
}

static void
<%=type_name%>_free(void* ptr)
{
    cumo_narray_data_t *na = (cumo_narray_data_t*)ptr;

    assert(na->base.type == CUMO_NARRAY_DATA_T);

    if (na->ptr != NULL) {
        cumo_cuda_runtime_free(na->ptr);
        na->ptr = NULL;
    }
    if (na->base.size > 0) {
        if (na->base.shape != NULL && na->base.shape != &(na->base.size)) {
            xfree(na->base.shape);
            na->base.shape = NULL;
        }
    }
    xfree(na);
}

static cumo_narray_type_info_t <%=type_name%>_info = {
  <% if is_bit %>
    1,             // element_bits
    0,             // element_bytes
    1,             // element_stride (in bits)
  <% else %>
    0,             // element_bits
    sizeof(dtype), // element_bytes
    sizeof(dtype), // element_stride (in bytes)
  <% end %>
};

<% if is_object %>
static void
<%=type_name%>_gc_mark(void *ptr)
{
    size_t n, i;
    VALUE *a;
    cumo_narray_data_t *na = ptr;

    if (na->ptr) {
        a = (VALUE*)(na->ptr);
        n = na->base.size;
        for (i=0; i<n; i++) {
            rb_gc_mark(a[i]);
        }
    }
}

static const rb_data_type_t <%=type_name%>_data_type = {
    "<%=full_class_name%>",
    {<%=type_name%>_gc_mark, <%=type_name%>_free, <%=type_name%>_memsize,},
    &cumo_na_data_type,
    &<%=type_name%>_info,
    0, // flags
};

<% else %>

static const rb_data_type_t <%=type_name%>_data_type = {
    "<%=full_class_name%>",
    {0, <%=type_name%>_free, <%=type_name%>_memsize,},
    &cumo_na_data_type,
    &<%=type_name%>_info,
    0, // flags
};

<% end %>

static VALUE
<%=c_func(0)%>(VALUE klass)
{
    cumo_narray_data_t *na = ALLOC(cumo_narray_data_t);

    na->base.ndim = 0;
    na->base.type = CUMO_NARRAY_DATA_T;
    na->base.flag[0] = CUMO_NA_FL0_INIT;
    na->base.flag[1] = CUMO_NA_FL1_INIT;
    na->base.size = 0;
    na->base.shape = NULL;
    na->base.reduce = INT2FIX(0);
    na->ptr = NULL;
    return TypedData_Wrap_Struct(klass, &<%=type_name%>_data_type, (void*)na);
}
