#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

struct cumo_<%=type_name%>_sum_impl {
    __device__ <%=dtype%> Identity() { return m_zero; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum += next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_prod_impl {
    __device__ <%=dtype%> Identity() { return m_one; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, <%=dtype%>& accum) { accum *= next; }
    __device__ <%=dtype%> MapOut(<%=dtype%> accum) { return accum; }
};

struct cumo_<%=type_name%>_min_impl {
    __device__ dtype Identity() { return DATA_MAX; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? next : accum; }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

struct cumo_<%=type_name%>_max_impl {
    __device__ dtype Identity() { return DATA_MIN; }
    __device__ dtype MapIn(dtype in, int64_t /*index*/) { return in; }
    __device__ void Reduce(dtype next, dtype& accum) { accum = next < accum ? accum : next; }
    __device__ dtype MapOut(dtype accum) { return accum; }
};

// TODO(sonots): Implement minmax
__global__ void cumo_<%=type_name%>_ptp_kernel(na_reduction_arg_t arg)
{
    dtype min=0,max=1;
    //<%=type_name%>_minmax_kernel<<<1,1>>>(n,p1,s1,&min,&max);
    char* p2 = arg.out.ptr;
    *(dtype*)p2 = m_sub(max,min);
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, cumo_<%=type_name%>_prod_impl>(*arg, cumo_<%=type_name%>_prod_impl{});
}

void cumo_<%=type_name%>_min_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, dtype, cumo_<%=type_name%>_min_impl>(*arg, cumo_<%=type_name%>_min_impl{});
}

void cumo_<%=type_name%>_max_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, dtype, cumo_<%=type_name%>_max_impl>(*arg, cumo_<%=type_name%>_max_impl{});
}

void cumo_<%=type_name%>_ptp_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_<%=type_name%>_ptp_kernel<<<1,1>>>(*arg);
}
