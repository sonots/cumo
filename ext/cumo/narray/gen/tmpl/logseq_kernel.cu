<% unless is_object %>
<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, cumo_na_indexer_t indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = f_seq(beg,step,c+i);
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = m_pow(base,x);
    }
}
<% end %>
<% end %>

<% layouts.each do |pfx, atype, at| %>
void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*indexer,beg,step,base,c);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_#{pfx}kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*indexer,beg,step,base,c);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
<% end %>
