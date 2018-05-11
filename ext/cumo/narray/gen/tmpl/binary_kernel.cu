<% unless type_name == 'robject' %>

<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(na_iarray_t a1, na_iarray_t a2, na_iarray_t a3, na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        *(dtype*)(p3) = m_<%=name%>(*(dtype*)(p1),*(dtype*)(p2));
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(na_iarray_t* a1, na_iarray_t* a2, na_iarray_t* a3, na_indexer_t* indexer)
{
    size_t gridDim = get_gridDim(indexer->total_size);
    size_t blockDim = get_blockDim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<gridDim, blockDim>>>(*a1,*a2,*a3,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<gridDim, blockDim>>>(*a1,*a2,*a3,*indexer);
        break;
    }
}
<% end %>
