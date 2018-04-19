static void CUDART_CB
<%=c_func(:nodef)%>_stream_callback(cudaStream_t stream, cudaError_t status, void *data)
{
    xfree(data);
}

static VALUE
<%=c_func(:nodef)%>(dtype x)
{
    VALUE v;
    dtype *ptr;

    v = nary_new(cT, 0, NULL);
    ptr = (dtype*)na_get_pointer_for_write(v);

    // To copy stack value into cuda memory asynchronously, we do
    // 1. copy to heap
    // 2. cudaMemcpyAsync from heap to cuda memory
    // 3. run callback to free the heap memory after memcpy finished
    //
    // FYI: We may have to care of cuda stream callback serializes stream execution when we support stream.
    // https://devtalk.nvidia.com/default/topic/822942/why-does-cudastreamaddcallback-serialize-kernel-execution-and-break-concurrency-/
    {
        cudaStream_t stream = 0;
        dtype* heap_x = ALLOC(dtype);
        *heap_x = x;
        cumo_cuda_runtime_check_status(cudaMemcpyAsync(ptr, heap_x, sizeof(dtype), cudaMemcpyHostToDevice, stream));
        cumo_cuda_runtime_check_status(cudaStreamAddCallback(stream, <%=c_func(:nodef)%>_stream_callback, heap_x, 0));
    }

    na_release_lock(v);
    return v;
}
