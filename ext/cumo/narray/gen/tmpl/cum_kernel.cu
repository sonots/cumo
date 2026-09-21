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

// base[i * step] as an iterator, so an operand that walks one stride is scanned
// where it lies however it walks. A flat operand takes the plain pointer
// instead, which is what lets thrust read it wide.
struct <%="cumo_thrust_#{name}_stride"%>
{
    int64_t step;
    __host__ __device__ int64_t operator()(int64_t i) const { return i * step; }
};

typedef thrust::transform_iterator<<%="cumo_thrust_#{name}_stride"%>,
        thrust::counting_iterator<int64_t> > <%="cumo_#{type_name}_#{name}_step_it"%>;
typedef thrust::permutation_iterator<thrust::device_ptr<dtype>,
        <%="cumo_#{type_name}_#{name}_step_it"%> > <%="cumo_#{type_name}_#{name}_strided_it"%>;

static <%="cumo_#{type_name}_#{name}_strided_it"%>
<%="cumo_#{type_name}_#{name}_strided"%>(dtype* base, int64_t step)
{
    <%="cumo_thrust_#{name}_stride"%> at;
    at.step = step;
    return thrust::make_permutation_iterator(
            thrust::device_pointer_cast(base),
            thrust::make_transform_iterator(thrust::make_counting_iterator<int64_t>(0), at));
}

// An operand no single stride reaches is copied into a buffer laid out end to
// end, and put back afterwards, which is two passes over the data against one
// launch a row. The address is taken with the accessor for the rank at hand:
// the general one divides by a run time extent per axis per element, and that
// costs more than the copy it is paying for.
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{type_name}_#{name}_gather_kernel_dim#{idim}"%>(cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, dtype* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        buf[i] = *(dtype*)cumo_na_iarray_stridx_at_dim<%=idim%>(&a, &indexer);
    }
}

__global__ void <%="cumo_#{type_name}_#{name}_scatter_kernel_dim#{idim}"%>(cumo_na_iarray_stridx_t a, cumo_na_indexer_t indexer, const dtype* buf) {
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        *(dtype*)cumo_na_iarray_stridx_at_dim<%=idim%>(&a, &indexer) = buf[i];
    }
}
<% end %>

static void
<%="cumo_#{type_name}_#{name}_gather_launch"%>(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, dtype* buf,
        size_t grid_dim, size_t block_dim)
{
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{type_name}_#{name}_gather_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, buf);
        break;
    <% end %>
    default:
        <%="cumo_#{type_name}_#{name}_gather_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, buf);
        break;
    }
}

static void
<%="cumo_#{type_name}_#{name}_scatter_launch"%>(cumo_na_iarray_stridx_t* a, cumo_na_indexer_t* indexer, const dtype* buf,
        size_t grid_dim, size_t block_dim)
{
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{type_name}_#{name}_scatter_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, buf);
        break;
    <% end %>
    default:
        <%="cumo_#{type_name}_#{name}_scatter_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a, *indexer, buf);
        break;
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
        thrust::inclusive_scan_by_key(thrust::cuda::par(alloc).on(cumo_cuda_stream()),
            keys, keys + (last - first),
            thrust::make_transform_iterator(first, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_output_iterator(result, <%="cumo_thrust_#{name}_narrow"%>()),
            thrust::equal_to<uint64_t>(), <%="cumo_thrust_#{name}#{j}"%>());
<% else %>
        thrust::inclusive_scan_by_key(thrust::cuda::par(alloc).on(cumo_cuda_stream()),
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
        thrust::inclusive_scan(thrust::cuda::par(alloc).on(cumo_cuda_stream()),
            thrust::make_transform_iterator(first, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_iterator(last, <%="cumo_thrust_#{name}_widen"%>()),
            thrust::make_transform_output_iterator(result, <%="cumo_thrust_#{name}_narrow"%>()),
            <%="cumo_thrust_#{name}#{j}"%>());
<% else %>
        thrust::inclusive_scan(thrust::cuda::par(alloc).on(cumo_cuda_stream()), first, last, result, <%="cumo_thrust_#{name}#{j}"%>());
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

// One row needs no key, and the key costs a division an element.
template<typename Iterator1, typename Iterator2>
static cudaError_t <%="cumo_#{type_name}_#{name}#{j}_scan_rows"%>(Iterator1 first, Iterator2 result, uint64_t total, uint64_t row_len)
{
    if (row_len == total) {
        return <%="cumo_#{type_name}_#{name}#{j}_scan"%>(first, first + total, result);
    }
    return <%="cumo_#{type_name}_#{name}#{j}_scan_by_key"%>(first, first + total, result, row_len);
}

template<typename Iterator1>
static cudaError_t <%="cumo_#{type_name}_#{name}#{j}_scan_into"%>(Iterator1 first, dtype* out, int64_t step_out, uint64_t total, uint64_t row_len)
{
    if (step_out == 1) {
        return <%="cumo_#{type_name}_#{name}#{j}_scan_rows"%>(first, thrust::device_pointer_cast(out), total, row_len);
    }
    return <%="cumo_#{type_name}_#{name}#{j}_scan_rows"%>(first, <%="cumo_#{type_name}_#{name}_strided"%>(out, step_out), total, row_len);
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

cudaError_t <%="cumo_#{type_name}_#{name}#{j}_batched_kernel_launch"%>(
        cumo_na_iarray_stridx_t* a_in, cumo_na_iarray_stridx_t* a_out,
        cumo_na_indexer_t* indexer, uint64_t row_len, ssize_t step_in, ssize_t step_out)
{
    uint64_t total = indexer->total_size;
    size_t grid_dim, block_dim;
    dtype *buf_in, *buf_out, *tmp = 0;
    int scatter_back = (step_out == 0);
    cudaError_t status;

    if (total == 0) { return cudaSuccess; }
    grid_dim = cumo_get_grid_dim(total);
    block_dim = cumo_get_block_dim(total);

    // One buffer covers whichever side no single stride reaches: the gather
    // fills it, the scan may read and write it in place, and the scatter empties
    // it. Two of them would put three copies of the array on the device where
    // the loop this replaces held one row.
    buf_in = (dtype*)a_in->ptr;
    buf_out = (dtype*)a_out->ptr;
    if (step_in == 0 || scatter_back) {
        tmp = (dtype*)cumo_cuda_runtime_malloc(sizeof(dtype) * total);
        if (step_in == 0) {
            <%="cumo_#{type_name}_#{name}_gather_launch"%>(a_in, indexer, tmp, grid_dim, block_dim);
            cumo_check_launch_holding(tmp);
            buf_in = tmp;
            step_in = 1;
        }
        if (scatter_back) { buf_out = tmp; step_out = 1; }
    }

    if (step_in == 1) {
        status = <%="cumo_#{type_name}_#{name}#{j}_scan_into"%>(
                thrust::device_pointer_cast(buf_in), buf_out, step_out, total, row_len);
    } else {
        status = <%="cumo_#{type_name}_#{name}#{j}_scan_into"%>(
                <%="cumo_#{type_name}_#{name}_strided"%>(buf_in, step_in), buf_out, step_out, total, row_len);
    }
    cumo_check_status_holding(status, tmp);

    if (scatter_back) {
        <%="cumo_#{type_name}_#{name}_scatter_launch"%>(a_out, indexer, tmp, grid_dim, block_dim);
        cumo_check_launch_holding(tmp);
    }
    if (tmp) { cumo_cuda_runtime_free((char*)tmp); }
    return cudaGetLastError();
}

<% end %>
<% end %>
