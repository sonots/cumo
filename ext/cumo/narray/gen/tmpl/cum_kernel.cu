<% unless type_name == 'robject' %>
<% widen = !acc_type.empty? && name == 'cumsum' %>
<% cum_t = widen ? acc_type : 'dtype' %>
<% if widen %>

// A running sum in the element type stops moving once the partial outgrows it,
// so the scan carries the accumulator type and each element is rounded as it is
// written.
struct <%="cumo_thrust_#{name}_widen"%>
{
    __host__ __device__ <%=acc_type%> operator()(dtype x) const { return <%=to_acc%>(x); }
};

struct <%="cumo_thrust_#{name}_narrow"%>
{
    __host__ __device__ dtype operator()(<%=acc_type%> x) const { return <%=from_acc%>(x); }
};
<% end %>

<% (is_float ? ["","_nan"] : [""]).each do |j| %>

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

<% if j == "" %>
// Which row an element belongs to. A scan by key needs one of these per
// element, and it is what lets every row go in a single launch.
struct <%="cumo_thrust_#{name}_row"%>
{
    uint64_t row_len;
    __host__ __device__ uint64_t operator()(uint64_t i) const { return i / row_len; }
};

// A row that is not laid out end to end is copied into a buffer that is, and
// put back afterwards, which is two passes over the data against one launch a
// row.
template <typename Iarray>
__global__ void <%="cumo_#{type_name}_#{name}_gather_kernel"%>(Iarray a, cumo_na_indexer_t indexer, dtype* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        buf[i] = *(dtype*)cumo_na_iarray_stridx_at_dim(&a, &indexer);
    }
}

template <typename Iarray>
__global__ void <%="cumo_#{type_name}_#{name}_scatter_kernel"%>(Iarray a, cumo_na_indexer_t indexer, const dtype* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim(&indexer, i);
        *(dtype*)cumo_na_iarray_stridx_at_dim(&a, &indexer) = buf[i];
    }
}
<% end %>

// Reusing the macro keeps the operator identical to the host loop, including
// how it carries a NaN. What does change is the association: a parallel scan
// does not add strictly left to right, so a float result can differ in the
// last ulp, the same way sum already does.
struct <%="cumo_thrust_#{name}#{j}"%>
{
    using first_argument_type  = <%=cum_t%>;
    using second_argument_type = <%=cum_t%>;
    using result_type          = <%=cum_t%>;
    __host__ __device__ <%=cum_t%> operator()(<%=cum_t%> x, <%=cum_t%> y) const { m_<%=name%><%=j%>(x,y); return x; }
};


// Nothing may reach the C caller: it has no handler, so an escaping exception
// is std::terminate. The status goes back instead and the caller raises.
// Every row in one call, so a loop over many rows costs no more launches than
// one row does. The key tells the scan where each row ends.
template<typename Iterator1, typename Iterator2>
static cudaError_t <%="cumo_#{type_name}_#{name}#{j}_scan_by_key"%>(Iterator1 first, Iterator1 last, Iterator2 result, uint64_t row_len)
{
    cumo_thrust_pool_allocator alloc;
    <%="cumo_thrust_#{name}_row"%> row = {row_len};
    auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator<uint64_t>(0), row);
    try {
<% if widen %>
        thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
            keys, keys + (last - first),
            thrust::make_transform_iterator(first, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_output_iterator(result, <%="cumo_thrust_#{name}_narrow"%>()),
            thrust::equal_to<uint64_t>(), <%="cumo_thrust_#{name}#{j}"%>());
<% else %>
        thrust::inclusive_scan_by_key(thrust::cuda::par(alloc),
            keys, keys + (last - first), first, result,
            thrust::equal_to<uint64_t>(), <%="cumo_thrust_#{name}#{j}"%>());
<% end %>
    } catch (const thrust::system_error& e) {
        return (cudaError_t)e.code().value();
    } catch (const std::bad_alloc&) {
        return cudaErrorMemoryAllocation;
    } catch (...) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

template<typename Iterator1, typename Iterator2>
static cudaError_t <%="cumo_#{type_name}_#{name}#{j}_scan"%>(Iterator1 first, Iterator1 last, Iterator2 result)
{
    cumo_thrust_pool_allocator alloc;
    try {
<% if widen %>
        thrust::inclusive_scan(thrust::cuda::par(alloc),
            thrust::make_transform_iterator(first, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_iterator(last, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_output_iterator(result, <%="cumo_thrust_#{name}_narrow"%>()),
            <%="cumo_thrust_#{name}#{j}"%>());
<% else %>
        thrust::inclusive_scan(thrust::cuda::par(alloc), first, last, result, <%="cumo_thrust_#{name}#{j}"%>());
<% end %>
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

cudaError_t <%="cumo_#{type_name}_#{name}#{j}_batched_kernel_launch"%>(
        cumo_na_iarray_stridx_t* a_in, cumo_na_iarray_stridx_t* a_out,
        cumo_na_indexer_t* indexer, uint64_t row_len, int flat_in, int flat_out)
{
    uint64_t total = indexer->total_size;
    size_t grid_dim, block_dim;
    dtype *buf_in, *buf_out, *tmp = 0;
    cudaError_t status;

    if (total == 0) { return cudaSuccess; }
    grid_dim = cumo_get_grid_dim(total);
    block_dim = cumo_get_block_dim(total);

    // One buffer covers whichever side is not laid out end to end: the gather
    // fills it, the scan may read and write it in place, and the scatter empties
    // it. Two of them would put three copies of the array on the device where
    // the loop this replaces held one row.
    buf_in = (dtype*)a_in->ptr;
    buf_out = (dtype*)a_out->ptr;
    if (!flat_in || !flat_out) {
        tmp = (dtype*)cumo_cuda_runtime_malloc(sizeof(dtype) * total);
        if (!flat_in) {
            <%="cumo_#{type_name}_#{name}_gather_kernel"%><<<grid_dim, block_dim>>>(*a_in, *indexer, tmp);
            cumo_check_launch_holding(tmp);
            buf_in = tmp;
        }
        if (!flat_out) { buf_out = tmp; }
    }

    {
        thrust::device_ptr<dtype> first = thrust::device_pointer_cast(buf_in);
        thrust::device_ptr<dtype> result = thrust::device_pointer_cast(buf_out);
        // One row needs no key, and the key costs a division an element.
        if (row_len == total) {
            status = <%="cumo_#{type_name}_#{name}#{j}_scan"%>(first, first + total, result);
        } else {
            status = <%="cumo_#{type_name}_#{name}#{j}_scan_by_key"%>(first, first + total, result, row_len);
        }
    }
    cumo_check_status_holding(status, tmp);

    if (!flat_out) {
        <%="cumo_#{type_name}_#{name}_scatter_kernel"%><<<grid_dim, block_dim>>>(*a_out, *indexer, tmp);
        cumo_check_launch_holding(tmp);
    }
    if (tmp) { cumo_cuda_runtime_free((char*)tmp); }
    return cudaGetLastError();
}

<% end %>
<% end %>
