<% unless defined?($cumo_narray_gen_tmpl_accum_binary_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_binary_kernel_included = 1 %>

<% unless type_name == 'robject' %>
//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

template<typename Iterator1, typename Iterator2>
__global__ void <%="cumo_#{type_name}_mulsum#{nan}_reduce_kernel"%>(Iterator1 p1_begin, Iterator1 p1_end, Iterator2 p2_begin, dtype* p3)
{
    dtype init = m_zero;
    *p3 = thrust::inner_product(thrust::cuda::par, p1_begin, p1_end, p2_begin, init, thrust_plus(), thrust_multiplies<%= "_mulsum#{nan}" unless nan.empty? %>());
}

__global__ void <%="cumo_#{type_name}_mulsum#{nan}_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        m_<%=name%><%=nan%>(*(dtype*)(p1+(i*s1)), *(dtype*)(p2+(i*s2)), *(dtype*)(p3+(i*s3)));
    }
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

void <%="cumo_#{type_name}_mulsum#{nan}_reduce_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, uint64_t n)
{
    ssize_t s1_idx = s1 / sizeof(dtype);
    ssize_t s2_idx = s2 / sizeof(dtype);
    thrust::device_ptr<dtype> p1_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> p1_end   = thrust::device_pointer_cast(((dtype*)p1) + n * s1_idx);
    thrust::device_ptr<dtype> p2_begin = thrust::device_pointer_cast((dtype*)p2);
    thrust::device_ptr<dtype> p2_end   = thrust::device_pointer_cast(((dtype*)p2) + n * s2_idx);
    if (s1_idx > 1 || s2_idx > 1) {
        thrust_strided_range<thrust::device_vector<dtype>::iterator> r1(p1_begin, p1_end, s1_idx);
        thrust_strided_range<thrust::device_vector<dtype>::iterator> r2(p2_begin, p2_end, s2_idx);
        <%="cumo_#{type_name}_mulsum#{nan}_reduce_kernel"%><<<1,1>>>(r1.begin(), r1.end(), r2.begin(), (dtype*)p3);
    } else {
        // ref. https://github.com/thrust/thrust/blob/master/examples/cuda/async_reduce.cu
        <%="cumo_#{type_name}_mulsum#{nan}_reduce_kernel"%><<<1,1>>>(p1_begin, p1_end, p2_begin, (dtype*)p3);
    }
}

void <%="cumo_#{type_name}_mulsum#{nan}_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{type_name}_mulsum#{nan}_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,n);
}
//<% end %>
<% end %>
<% end %>
