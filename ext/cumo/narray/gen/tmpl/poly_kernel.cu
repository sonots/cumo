<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t x, cumo_na_iarray_t* coef, int ncoef, cumo_na_iarray_t z, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        int k;
        dtype xv, y;
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        xv = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&x, &indexer);
        // No coefficient at all answers x itself, which is what the host loop did
        y = (ncoef > 0) ? *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&coef[ncoef - 1], &indexer) : xv;
        for (k = ncoef - 2; k >= 0; --k) {
            y = m_mul(xv, y);
            y = m_add(y, *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&coef[k], &indexer));
        }
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&z, &indexer) = y;
    }
}
<% end %>

// The coefficient count is whatever the caller passed, so the kernel reads the
// coefficients through a device array rather than a kernel argument.
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* x, cumo_na_iarray_t* coef, int ncoef, cumo_na_iarray_t* z, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    size_t bytes = sizeof(cumo_na_iarray_t) * (ncoef > 0 ? ncoef : 1);
    cumo_na_iarray_t* d_coef = (cumo_na_iarray_t*)cumo_cuda_runtime_malloc(bytes);

    if (ncoef > 0) {
        cudaMemcpyAsync(d_coef, coef, sizeof(cumo_na_iarray_t) * ncoef, cudaMemcpyHostToDevice, 0);
    }
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*x, d_coef, ncoef, *z, *indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*x, d_coef, ncoef, *z, *indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
    cumo_cuda_runtime_free((char*)d_coef);
}
<% end %>
