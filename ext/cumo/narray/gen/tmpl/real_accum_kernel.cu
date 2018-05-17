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

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_sum_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = m_zero;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::plus<dtype>());
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_prod_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = m_one;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::multiplies<dtype>());
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_min_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = DATA_MAX;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::minimum<dtype>());
}

template<typename Iterator1>
__global__ void cumo_<%=type_name%>_max_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = DATA_MIN;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::maximum<dtype>());
}

// TODO(sonots): Implement minmax
__global__ void cumo_<%=type_name%>_ptp_kernel(uint64_t n, char *p1, ssize_t s1, <%=dtype%>* p2)
{
    dtype min=0,max=1;
    //<%=type_name%>_minmax_kernel<<<1,1>>>(n,p1,s1,&min,&max);
    *p2 = m_sub(max,min);
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void cumo_<%=type_name%>_sum_kernel_launch(na_reduction_arg_t* arg)
{
    cumo_reduce<dtype, <%=dtype%>, <%=dtype%>, cumo_<%=type_name%>_sum_impl>(*arg, cumo_<%=type_name%>_sum_impl{});
}

void cumo_<%=type_name%>_prod_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_prod_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_prod_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void cumo_<%=type_name%>_min_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_min_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_min_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void cumo_<%=type_name%>_max_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        cumo_<%=type_name%>_max_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        cumo_<%=type_name%>_max_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void cumo_<%=type_name%>_ptp_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    cumo_<%=type_name%>_ptp_kernel<<<1,1>>>(n,p1,s1,(<%=dtype%>*)p2);
}
