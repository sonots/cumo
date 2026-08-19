<% unless c_iter.include? 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_bit_iarray_stridx_t a1, cumo_na_iarray_stridx_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p2 = cumo_na_iarray_stridx_at_dim<%=idim%>(&a2, &indexer);
        CUMO_BIT_DIGIT y = <%=macro%>(*(<%=dtype%>*)(p2));
        CUMO_STORE_BIT(a1.ptr, cumo_na_bit_iarray_stridx_at_dim<%=idim%>(&a1, &indexer), y);
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
