<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_bit_iarray_stridx_t a1, cumo_na_bit_iarray_stridx_t a3, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        CUMO_BIT_DIGIT x;
        CUMO_LOAD_BIT(a1.ptr, cumo_na_bit_iarray_stridx_at_dim<%=idim%>(&a1, &indexer), x);
        CUMO_STORE_BIT(a3.ptr, cumo_na_bit_iarray_stridx_at_dim<%=idim%>(&a3, &indexer), x);
    }
}
<% end %>

__global__ void <%="cumo_#{c_iter}_contiguous_kernel"%>(CUMO_BIT_DIGIT *a1, ssize_t o1, uint64_t w1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n, uint64_t w3)
{
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w3; w += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x = cumo_bit_gather_word(a1,o1,w1,w);
        cumo_bit_store_word(a3, w, x, p3, n);
    }
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_bit_iarray_stridx_t* a3, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a3,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a3,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_contiguous_kernel_launch"%>(CUMO_BIT_DIGIT *a1, size_t p1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n)
{
    ssize_t o1 = (ssize_t)p1 - (ssize_t)p3;
    uint64_t w1 = (p1 + n + CUMO_NB - 1) / CUMO_NB;
    uint64_t w3 = (p3 + n + CUMO_NB - 1) / CUMO_NB;
    size_t grid_dim = cumo_get_grid_dim(w3);
    size_t block_dim = cumo_get_block_dim(w3);
    <%="cumo_#{c_iter}_contiguous_kernel"%><<<grid_dim, block_dim>>>(a1,o1,w1,a3,p3,n,w3);
    cumo_cuda_runtime_check_kernel_launch();
}
