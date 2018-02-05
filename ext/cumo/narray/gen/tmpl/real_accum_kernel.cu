template<typename Iterator1, typename Iterator2>
__global__ void <%=type_name%>_sum_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = m_zero;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::plus<dtype>());
}

void <%=type_name%>_sum_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p) + n * s1_idx);
    if (s1_idx == 1) {
        <%=type_name%>_sum_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        <%=type_name%>_sum_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

template<typename Iterator1, typename Iterator2>
__global__ void <%=type_name%>_prod_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = m_one;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::multiplies<dtype>());
}

void <%=type_name%>_prod_kernel_launch(uint64_t n, char *p1, ssize_t s1, uint64_t n, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        <%=type_name%>_prod_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        <%=type_name%>_prod_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

template<typename Iterator1, typename Iterator2>
__global__ void <%=type_name%>_min_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = DATA_MAX;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::minimum<dtype>());
}

void <%=type_name%>_min_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        <%=type_name%>_min_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        <%=type_name%>_min_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

template<typename Iterator1, typename Iterator2>
__global__ void <%=type_name%>_max_kernel(Iterator1 p1_begin, Iterator1 p1_end, <%=dtype%>* p2)
{
    dtype init = DATA_MIN;
    *p2 = thrust::reduce(thrust::cuda::par, p1_begin, p1_end, init, thrust::maximum<dtype>());
}

void <%=type_name%>_max_kernel_launch(uint64_t n, char *p1, ssize_t stride, char *p2)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    thrust::device_ptr<dtype> data_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> data_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    if (s1_idx == 1) {
        <%=type_name%>_max_kernel<<<1,1>>>(data_begin, data_end, (<%=dtype%>*)p2);
    } else {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> range(data_begin, data_end, s1_idx);
        <%=type_name%>_max_kernel<<<1,1>>>(range.begin(), range.end(), (<%=dtype%>*)p2);
    }
}

void <%=type_name%>_minmax_kernel_launch(uint64_t n, char *p1, ssize_t s1, dtype* amin, dtype* amax);

__global__ void <%=type_name%>_ptp_kernel(uint64_t n, char *p1, ssize_t s1, <%=dtype%>* p2)
{
    dtype min,max;
    <%=type_name%>_minmax_kernel_launch(n,p1,s1,&min,&max);
    *p2 = m_sub(max,min);
}

void <%=type_name%>_ptp_kernel_launch(uint64_t n, char *p1, ssize_t s1, char *p2)
{
    <%=type_name%>_ptp_kernel<<<1,1>>>(n,p1,s1,(<%=dtype%>*)p2);
}
