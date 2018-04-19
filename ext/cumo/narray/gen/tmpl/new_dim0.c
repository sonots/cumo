void <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(dtype *ptr, dtype x);

static VALUE
<%=c_func(:nodef)%>(dtype x)
{
    VALUE v;
    dtype *ptr;

    v = nary_new(cT, 0, NULL);
    ptr = (dtype*)na_get_pointer_for_write(v);
    <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(ptr, x);

    na_release_lock(v);
    return v;
}
