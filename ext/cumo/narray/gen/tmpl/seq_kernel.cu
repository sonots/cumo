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
<% layouts = [["", "cumo_na_iarray_t", "cumo_na_iarray_at_dim"], ["stridx_", "cumo_na_iarray_stridx_t", "cumo_na_iarray_stridx_at_dim"]] %>
<% layouts.each do |pfx, atype, at| %>
<% indexer_dims(pfx.empty?).each do |idim| %>
__global__ void <%="cumo_#{c_iter}_#{pfx}kernel_dim#{idim}"%>(<%=atype%> a1, cumo_na_indexer_t indexer, seq_data_t beg, seq_data_t step, seq_count_t c)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
<% if type_name == 'uint64' %>
        // UInt64 is the one type the signed 64-bit step cannot carry on its
        // own: it wraps a negative start the way fill and cast do, but clamps
        // everything from 2**63 up, which UInt64 holds exactly.
        seq_data_t v = f_seq(beg,step,c+i);
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = v < 0 ? (dtype)(int64_t)v : (dtype)v;
<% else %>
        // f_seq answers a double for the integer types, and the device
        // saturates an out-of-range double where fill and cast wrap, so a
        // negative start collapsed to zero. A signed 64-bit step in between
        // keeps seq answering what they answer.
        *(dtype*)<%=at%><%=idim%>(&a1, &indexer) = <% if is_int %>(dtype)(int64_t)<% end %>f_seq(beg,step,c+i);
<% end %>
    }
}
<% end %>
<% end %>

<% layouts.each do |pfx, atype, at| %>
void <%="cumo_#{c_iter}_#{pfx}kernel_launch"%>(<%=atype%>* a1, cumo_na_indexer_t* indexer, seq_data_t beg, seq_data_t step, seq_count_t c)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_#{pfx}kernel", "*a1,*indexer,beg,step,c", narrow: (%w[a1] if pfx.empty?)) %>
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
<% end %>
