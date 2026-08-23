<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_bit_iarray_t a2, cumo_na_indexer_t indexer, uint64_t end, CUMO_BIT_DIGIT *out)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT b = 0;
        if (i < indexer.total_size) {
            cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
            dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
            b = (m_<%=name%>(x)) ? 1:0;
        }
        if (out) {
            cumo_bit_store_ballot(out, i, b, indexer.total_size);
        } else {
            CUMO_STORE_BIT(a2.ptr, cumo_na_bit_iarray_at_dim<%=idim%>(&a2, &indexer), b);
        }
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_bit_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    // A contiguous output gives each warp a whole word of it, which the warp
    // builds with a ballot and stores once. Anything else keeps the atomic,
    // since the lanes then hold bits of no one word.
    CUMO_BIT_DIGIT *out = cumo_na_bit_iarray_is_flat(a2, indexer) ? a2->ptr + a2->pos / CUMO_NB : NULL;
    uint64_t end = out ? ((indexer->total_size + CUMO_NB - 1) & ~(uint64_t)(CUMO_NB - 1)) : indexer->total_size;
    size_t grid_dim = cumo_get_grid_dim(end);
    // the ballot needs whole warps, so a short loop cannot shrink the block
    size_t block_dim = out ? CUMO_MAX_BLOCK_DIM : cumo_get_block_dim(end);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer,end,out);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer,end,out);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
