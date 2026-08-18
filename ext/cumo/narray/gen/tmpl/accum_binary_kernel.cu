<% unless defined?($cumo_narray_gen_tmpl_accum_binary_kernel_included) %>
<% $cumo_narray_gen_tmpl_accum_binary_kernel_included = 1 %>

<% unless type_name == 'robject' %>
// One output at a time, so the reduce is a plain block tree over n elements of
// two operands. Enough blocks to fill the device, then one block folds their
// partials and adds the result to whatever the init argument left in the output.
#define CUMO_MULSUM_BLOCK_DIM 512
#define CUMO_MULSUM_MAX_GRID_DIM 1024

//<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

__device__ static void <%="cumo_#{type_name}_#{name}#{nan}_block_reduce"%>(dtype *sdata, dtype acc)
{
    sdata[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int stride = CUMO_MULSUM_BLOCK_DIM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] = m_add(sdata[threadIdx.x], sdata[threadIdx.x + stride]);
        }
        __syncthreads();
    }
}

// Writes its block's partial to partial[blockIdx.x], or straight to the output
// when the grid is one block and there is nothing left to combine.
__global__ void <%="cumo_#{type_name}_#{name}#{nan}_partial_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, uint64_t n, dtype *partial)
{
    __shared__ dtype sdata[CUMO_MULSUM_BLOCK_DIM];
    dtype acc = m_zero;

    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dtype x = *(dtype*)(p1 + i * s1);
        dtype y = *(dtype*)(p2 + i * s2);
        m_<%=name%><%=nan%>(x, y, acc);
    }
    <%="cumo_#{type_name}_#{name}#{nan}_block_reduce"%>(sdata, acc);
    if (threadIdx.x == 0) {
        if (partial) {
            partial[blockIdx.x] = sdata[0];
        } else {
            *(dtype*)p3 = m_add(*(dtype*)p3, sdata[0]);
        }
    }
}

__global__ void <%="cumo_#{type_name}_#{name}#{nan}_combine_kernel"%>(char *p3, dtype *partial, uint64_t n)
{
    __shared__ dtype sdata[CUMO_MULSUM_BLOCK_DIM];
    dtype acc = m_zero;

    for (uint64_t i = threadIdx.x; i < n; i += CUMO_MULSUM_BLOCK_DIM) {
        acc = m_add(acc, partial[i]);
    }
    <%="cumo_#{type_name}_#{name}#{nan}_block_reduce"%>(sdata, acc);
    if (threadIdx.x == 0) {
        *(dtype*)p3 = m_add(*(dtype*)p3, sdata[0]);
    }
}

__global__ void <%="cumo_#{type_name}_#{name}#{nan}_kernel"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        m_<%=name%><%=nan%>(*(dtype*)(p1+(i*s1)), *(dtype*)(p2+(i*s2)), *(dtype*)(p3+(i*s3)));
    }
}

void <%="cumo_#{type_name}_#{name}#{nan}_reduce_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, uint64_t n)
{
    uint64_t grid_dim = (n + CUMO_MULSUM_BLOCK_DIM - 1) / CUMO_MULSUM_BLOCK_DIM;
    dtype *partial = NULL;

    if (grid_dim == 0) grid_dim = 1;
    if (grid_dim > CUMO_MULSUM_MAX_GRID_DIM) grid_dim = CUMO_MULSUM_MAX_GRID_DIM;
    if (grid_dim > 1) {
        partial = (dtype*)cumo_cuda_runtime_malloc(sizeof(dtype) * grid_dim);
    }
    <%="cumo_#{type_name}_#{name}#{nan}_partial_kernel"%><<<grid_dim, CUMO_MULSUM_BLOCK_DIM>>>(p1,p2,p3,s1,s2,n,partial);
    if (partial) {
        <%="cumo_#{type_name}_#{name}#{nan}_combine_kernel"%><<<1, CUMO_MULSUM_BLOCK_DIM>>>(p3,partial,grid_dim);
        cumo_cuda_runtime_free((char*)partial);
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{type_name}_#{name}#{nan}_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{type_name}_#{name}#{nan}_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n);
    cumo_cuda_runtime_check_kernel_launch();
}
//<% end %>
<% end %>
<% end %>
