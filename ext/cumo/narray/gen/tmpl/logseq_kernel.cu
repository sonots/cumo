<% unless is_object %>
<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, cumo_na_indexer_t indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, int exp_base)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = f_seq(beg,step,c+i);
<% if is_complex || !is_double_precision %>
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = m_pow(base,x);
<% else %>
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = exp_base == 10 ? exp10(x) : exp_base == 2 ? exp2(x) : m_pow(base,x);
<% end %>
    }
}
<% end %>
<% end %>

<% layouts.each do |pfx, atype, at| %>
void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_data_t base, seq_count_t c, int exp_base)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_#{pfx}kernel", "*a1,*indexer,beg,step,base,c,exp_base") %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
<% end %>
