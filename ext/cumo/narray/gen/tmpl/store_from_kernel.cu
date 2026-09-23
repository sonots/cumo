<% unless c_iter.include? 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)(p1) = <%=macro%>(*(<%=dtype%>*)(p2));
    }
}
<% end %>

<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_stridx_kernel_dim#{idim}"%>(cumo_na_iarray_stridx_t a1, cumo_na_iarray_stridx_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_stridx_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_stridx_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)(p1) = <%=macro%>(*(<%=dtype%>*)(p2));
    }
}
<% end %>

void <%="cumo_#{c_iter}_stridx_kernel_launch"%>(cumo_na_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_stridx_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_stridx_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

__global__ void <%="cumo_#{c_iter}_transpose_kernel"%>(char *p1, char *p2, uint64_t rows, uint64_t cols)
{
    CUMO_TRANSPOSE_TILE_DECL(dtype, tile);

    CUMO_TRANSPOSE_TILE_LOOP(tile, rows, cols,
        <%=macro%>(*(<%=dtype%>*)(p2 + cumo_tile_src * sizeof(<%=dtype%>))),
        *(dtype*)(p1 + cumo_tile_dst * sizeof(dtype)) = cumo_tile_val;
    );
}

// True when the destination runs along its rows and the source along its
// columns, which is what a copy of a transposed view looks like. A side shorter
// than a tile leaves most of each warp idle, and the plain loop already reads or
// writes such a side in one go, so the tiles only pay off once both sides reach
// one.
static int
<%="cumo_#{c_iter}_is_transpose"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    return CUMO_TRANSPOSE_TILE_FITS(indexer) &&
        a1->step[1] == (ssize_t)sizeof(dtype) &&
        a1->step[0] == (ssize_t)(sizeof(dtype) * indexer->shape[1]) &&
        a2->step[0] == (ssize_t)sizeof(<%=dtype%>) &&
        a2->step[1] == (ssize_t)(sizeof(<%=dtype%>) * indexer->shape[0]);
}

<% if name == parent.parent.name %>
#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif

<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
template<typename V>
__global__ void <%="cumo_#{c_iter}_vec_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        *(V*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer) = *(V*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
    }
}
<% end %>

template<typename V>
static void
<%="cumo_#{c_iter}_vec_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_vec_kernel_dim#{idim}"%><V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_vec_kernel_dim"%><V><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    }
}

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

// A copy between arrays of the same type moves bytes, so a run that is
// contiguous on both sides can go several elements to a thread. The address
// arithmetic, not the memory, bounds a strided copy one element at a time.
static int
<%="cumo_#{c_iter}_copy_wide"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    int last = indexer->ndim - 1, k;
    size_t w, bytes;
    cumo_na_iarray_t b1, b2;
    cumo_na_indexer_t ix;

    if (indexer->ndim == 0 || indexer->total_size == 0) { return 0; }
    if (a1->step[last] != (ssize_t)sizeof(dtype) || a2->step[last] != (ssize_t)sizeof(dtype)) { return 0; }
    bytes = indexer->shape[last] * sizeof(dtype);
    for (w = 16; w > sizeof(dtype); w /= 2) {
        if (bytes % w != 0 || (uintptr_t)a1->ptr % w != 0 || (uintptr_t)a2->ptr % w != 0) { continue; }
        for (k = 0; k < last; ++k) {
            if (indexer->shape[k] > 1 && (a1->step[k] % (ssize_t)w != 0 || a2->step[k] % (ssize_t)w != 0)) { break; }
        }
        if (k == last) { break; }
    }
    if (w <= sizeof(dtype)) { return 0; }

    b1 = *a1;
    b2 = *a2;
    ix = *indexer;
    b1.step[last] = b2.step[last] = (ssize_t)w;
    ix.shape[last] = bytes / w;
    ix.total_size = indexer->total_size / indexer->shape[last] * ix.shape[last];
    switch (w) {
    case 16: <%="cumo_#{c_iter}_vec_launch"%><uint4>(&b1, &b2, &ix); break;
    case 8: <%="cumo_#{c_iter}_vec_launch"%><uint2>(&b1, &b2, &ix); break;
    case 4: <%="cumo_#{c_iter}_vec_launch"%><uint32_t>(&b1, &b2, &ix); break;
    default: <%="cumo_#{c_iter}_vec_launch"%><uint16_t>(&b1, &b2, &ix); break;
    }
    return 1;
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim, block_dim;

    if (<%="cumo_#{c_iter}_is_transpose"%>(a1, a2, indexer)) {
        CUMO_TRANSPOSE_LAUNCH(<%="cumo_#{c_iter}_transpose_kernel"%>,
                              indexer->shape[0], indexer->shape[1],
                              a1->ptr, a2->ptr);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }
<% if name == parent.parent.name %>
    if (<%="cumo_#{c_iter}_copy_wide"%>(a1, a2, indexer)) {
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }
<% end %>

    grid_dim = cumo_get_grid_dim(indexer->total_size);
    block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
