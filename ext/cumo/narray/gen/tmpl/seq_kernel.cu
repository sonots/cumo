<% if is_int && !is_object %>
typedef double seq_data_t;
<% else %>
typedef dtype seq_data_t;
<% end %>

<% if is_object %>
typedef size_t seq_count_t;
<% else %>
typedef double seq_count_t;
<% end %>

<% unless is_object %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_indexer_t indexer, seq_data_t beg, seq_data_t step, seq_count_t c)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        // f_seq answers a double for the integer types, and the device
        // saturates an out-of-range double where fill and cast wrap, so a
        // negative start collapsed to zero. A signed 64-bit step in between
        // keeps seq answering what they answer.
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer) = <% if is_int %>(dtype)(int64_t)<% end %>f_seq(beg,step,c+i);
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_count_t c)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*indexer,beg,step,c);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*indexer,beg,step,c);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
