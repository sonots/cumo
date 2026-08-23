<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
// A Ruby numeric operand rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. It reaches the
// left side through coerce, so use_scalar says which side it is on. It is the
// same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_bit_iarray_t a3, cumo_na_indexer_t indexer, dtype sv, int use_scalar, uint64_t end, CUMO_BIT_DIGIT *out)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT b = 0;
        if (i < indexer.total_size) {
            cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
            dtype v = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
            dtype x, y;
            switch (use_scalar) {
            case 1:  x = v; y = sv; break;
            case 2:  x = sv; y = v; break;
            default: x = v; y = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
            }
            b = (m_<%=name%>(x,y)) ? 1:0;
        }
        if (out) {
            cumo_bit_store_ballot(out, i, b, indexer.total_size);
        } else {
            CUMO_STORE_BIT(a3.ptr, cumo_na_bit_iarray_at_dim<%=idim%>(&a3, &indexer), b);
        }
    }
}
<% end %>

static void <%="cumo_#{c_iter}_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_bit_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int use_scalar)
{
    // A contiguous output gives each warp a whole word of it, which the warp
    // builds with a ballot and stores once. Anything else keeps the atomic,
    // since the lanes then hold bits of no one word.
    CUMO_BIT_DIGIT *out = cumo_na_bit_iarray_is_flat(a3, indexer) ? a3->ptr + a3->pos / CUMO_NB : NULL;
    uint64_t end = out ? ((indexer->total_size + CUMO_NB - 1) & ~(uint64_t)(CUMO_NB - 1)) : indexer->total_size;
    size_t grid_dim = cumo_get_grid_dim(end);
    // the ballot needs whole warps, so a short loop cannot shrink the block
    size_t block_dim = out ? CUMO_MAX_BLOCK_DIM : cumo_get_block_dim(end);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar,end,out);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar,end,out);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_bit_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,a2,a3,indexer,sv,0);
}

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_bit_iarray_t* a3, cumo_na_indexer_t* indexer, int scalar_is_left)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,scalar_is_left ? 2 : 1);
}
<% end %>
