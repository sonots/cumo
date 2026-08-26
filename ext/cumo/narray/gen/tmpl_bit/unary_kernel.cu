<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_bit_iarray_stridx_t a1, cumo_na_bit_iarray_stridx_t a3, cumo_na_indexer_t indexer, uint64_t end, CUMO_BIT_DIGIT *out)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < end; i += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT y = 0;
        if (i < indexer.total_size) {
            cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
            CUMO_BIT_DIGIT x;
            CUMO_LOAD_BIT(a1.ptr, cumo_na_bit_iarray_stridx_at_dim<%=idim%>(&a1, &indexer), x);
            y = m_<%=name%>(x) & 1u;
        }
        if (out) {
            cumo_bit_store_ballot(out, i, y, indexer.total_size);
        } else {
            CUMO_STORE_BIT(a3.ptr, cumo_na_bit_iarray_stridx_at_dim<%=idim%>(&a3, &indexer), y);
        }
    }
}
<% end %>

// One thread per word of the output, taking the word from the operand's own
// rows. The lanes of a warp would otherwise each pay the indexer to read one
// bit of the same word.
__global__ void <%="cumo_#{c_iter}_run_kernel"%>(cumo_na_bit_iarray_stridx_t a1, cumo_na_indexer_t indexer, cumo_bit_run_t run, CUMO_BIT_DIGIT *out, uint64_t n, uint64_t w3)
{
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w3; w += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x = cumo_bit_run_word(&a1, &indexer, run, w, n);
        cumo_bit_store_word(out, w, m_<%=name%>(x), 0, n);
    }
}

__global__ void <%="cumo_#{c_iter}_contiguous_kernel"%>(CUMO_BIT_DIGIT *a1, ssize_t o1, uint64_t w1, CUMO_BIT_DIGIT *a3, size_t p3, uint64_t n, uint64_t w3)
{
    for (uint64_t w = blockIdx.x * blockDim.x + threadIdx.x; w < w3; w += blockDim.x * gridDim.x) {
        CUMO_BIT_DIGIT x = cumo_bit_gather_word(a1,o1,w1,w);
        cumo_bit_store_word(a3, w, m_<%=name%>(x), p3, n);
    }
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_iarray_stridx_t* a1, cumo_na_bit_iarray_stridx_t* a3, cumo_na_indexer_t* indexer)
{
    // Only the output decides: a warp holds one whole word of it however the
    // operand is laid out, so the lanes can pool their bits into that word
    // rather than each taking it with an atomic.
    CUMO_BIT_DIGIT *out = cumo_na_bit_iarray_stridx_is_flat(a3, indexer) ? a3->ptr + a3->pos / CUMO_NB : NULL;

    // With the output taking a whole word from one thread, an operand whose
    // rows run bit by bit can be gathered a word at a time as well.
    if (out != NULL) {
        cumo_bit_run_t run = cumo_bit_make_run(a1, indexer);
        if (run.ok) {
            uint64_t w3 = (indexer->total_size + CUMO_NB - 1) / CUMO_NB;
            <%="cumo_#{c_iter}_run_kernel"%><<<cumo_get_grid_dim(w3), cumo_get_block_dim(w3)>>>(
                *a1, *indexer, run, out, indexer->total_size, w3);
            cumo_cuda_runtime_check_kernel_launch();
            return;
        }
    }

    uint64_t end = out ? ((indexer->total_size + CUMO_NB - 1) & ~(uint64_t)(CUMO_NB - 1)) : indexer->total_size;
    size_t grid_dim = cumo_get_grid_dim(end);
    // the ballot needs whole warps, so a short loop cannot shrink the block
    size_t block_dim = out ? CUMO_MAX_BLOCK_DIM : cumo_get_block_dim(end);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a3,*indexer,end,out);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a3,*indexer,end,out);
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
