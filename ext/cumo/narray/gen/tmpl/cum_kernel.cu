<% unless type_name == 'robject' %>
<% (is_float ? ["","_nan"] : [""]).each do |j| %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

// Reusing the macro keeps the operator identical to the host loop, including
// how it carries a NaN. What does change is the association: a parallel scan
// does not add strictly left to right, so a float result can differ in the
// last ulp, the same way sum already does.
struct <%="cumo_thrust_#{name}#{j}"%>
{
    using first_argument_type  = dtype;
    using second_argument_type = dtype;
    using result_type          = dtype;
    __host__ __device__ dtype operator()(dtype x, dtype y) const { m_<%=name%><%=j%>(x,y); return x; }
};

// Nothing may reach the C caller: it has no handler, so an escaping exception
// is std::terminate. The status goes back instead and the caller raises.
template<typename Iterator1, typename Iterator2>
static cudaError_t <%="cumo_#{type_name}_#{name}#{j}_scan"%>(Iterator1 first, Iterator1 last, Iterator2 result)
{
    cumo_thrust_pool_allocator alloc;
    try {
        thrust::inclusive_scan(thrust::cuda::par(alloc), first, last, result, <%="cumo_thrust_#{name}#{j}"%>());
    } catch (const thrust::system_error& e) {
        return (cudaError_t)e.code().value();
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

cudaError_t <%="cumo_#{type_name}_#{name}#{j}_kernel_launch"%>(char *p1, char *p2, ssize_t s1, ssize_t s2, uint64_t n)
{
    ssize_t s1_idx = s1 / (ssize_t)sizeof(dtype);
    ssize_t s2_idx = s2 / (ssize_t)sizeof(dtype);
    thrust::device_ptr<dtype> p1_begin = thrust::device_pointer_cast((dtype*)p1);
    thrust::device_ptr<dtype> p2_begin = thrust::device_pointer_cast((dtype*)p2);
    cudaError_t status;

    // A broadcast operand has stride 0 and a reversed view a negative one, so
    // anything but 1 has to go through the strided range.
    if (s1_idx == 1 && s2_idx == 1) {
        status = <%="cumo_#{type_name}_#{name}#{j}_scan"%>(p1_begin, p1_begin + n, p2_begin);
    } else {
        typedef cumo_thrust_strided_range<thrust::device_vector<dtype>::iterator> range_t;
        range_t r1(p1_begin, (range_t::difference_type)s1_idx, (range_t::difference_type)n);
        range_t r2(p2_begin, (range_t::difference_type)s2_idx, (range_t::difference_type)n);
        status = <%="cumo_#{type_name}_#{name}#{j}_scan"%>(r1.begin(), r1.end(), r2.begin());
    }
    if (status != cudaSuccess) {
        return status;
    }
    return cudaGetLastError();
}
<% end %>
<% end %>
