static VALUE
<%=c_func(:nodef)%>(dtype x)
{
    VALUE v;
    dtype *ptr;

    v = nary_new(cT, 0, NULL);
    ptr = (dtype*)(char*)na_get_pointer_for_write(v);
    // TODO(sonots): Any ways to copy stack data into cumo memory without synchronize?
    cumo_cuda_runtime_check_status(cudaDeviceSynchronize());
    *ptr = x;
    na_release_lock(v);
    return v;
}
