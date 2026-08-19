<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_iarray_t a4, cumo_na_indexer_t indexer, int* invalid)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype min = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        dtype max = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        // Leaving the element unwritten keeps a scalar min > max, where every
        // element takes this branch, from touching the output before it raises.
        if (m_gt(min,max)) { *invalid = 1; continue; }
        if (m_lt(x,min)) { x = min; }
        if (m_gt(x,max)) { x = max; }
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a4, &indexer) = x;
    }
}

__global__ void <%="cumo_#{c_iter}_min_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype min = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer) = m_lt(x,min) ? min : x;
    }
}

__global__ void <%="cumo_#{c_iter}_max_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype max = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer) = m_gt(x,max) ? max : x;
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer, int* invalid)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*a4,*indexer,invalid);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*a4,*indexer,invalid);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_min_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_min_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_min_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_max_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_max_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_max_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
