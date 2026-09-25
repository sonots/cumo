<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% indexer_dims(pfx.empty?).each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, <%=atype%> a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = <%=at%><%=idim%>(&a1, &indexer);
        char* p2 = <%=at%><%=idim%>(&a2, &indexer);
        *(dtype*)(p1) = m_<%=name%>(*(dtype*)(p1), *(<%=dtype%>*)(p2));
    }
}
<% end %>

void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, <%=atype%>* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_#{pfx}kernel", "*a1,*a2,*indexer", narrow: (%w[a1 a2] if pfx.empty?)) %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
