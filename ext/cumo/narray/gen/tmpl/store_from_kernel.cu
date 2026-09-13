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
        <%="cumo_#{c_iter}_stridx_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_stridx_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

__global__ void <%="cumo_#{c_iter}_transpose_kernel"%>(char *p1, char *p2, uint64_t rows, uint64_t cols)
{
    __shared__ dtype tile[CUMO_TRANSPOSE_TILE][CUMO_TRANSPOSE_TILE + 1];
    uint64_t base;

    for (base = (uint64_t)blockIdx.y * CUMO_TRANSPOSE_TILE; base < rows;
         base += (uint64_t)gridDim.y * CUMO_TRANSPOSE_TILE) {
        uint64_t x = base + threadIdx.x;
        uint64_t y = (uint64_t)blockIdx.x * CUMO_TRANSPOSE_TILE + threadIdx.y;
        int j;

        __syncthreads();
        for (j = 0; threadIdx.y + j < CUMO_TRANSPOSE_TILE; j += CUMO_TRANSPOSE_ROWS) {
            if (x < rows && y + j < cols) {
                tile[threadIdx.y + j][threadIdx.x] =
                    <%=macro%>(*(<%=dtype%>*)(p2 + ((y + j) * rows + x) * sizeof(<%=dtype%>)));
            }
        }
        __syncthreads();
        x = (uint64_t)blockIdx.x * CUMO_TRANSPOSE_TILE + threadIdx.x;
        y = base + threadIdx.y;
        for (j = 0; threadIdx.y + j < CUMO_TRANSPOSE_TILE; j += CUMO_TRANSPOSE_ROWS) {
            if (x < cols && y + j < rows) {
                *(dtype*)(p1 + ((y + j) * cols + x) * sizeof(dtype)) = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }
}

// True when the destination runs along its rows and the source along its
// columns, which is what a copy of a transposed view looks like. A side shorter
// than a tile leaves most of each warp idle, and the plain loop already reads or
// writes such a side in one go, so the tiles only pay off once both sides reach
// one.
static int
<%="cumo_#{c_iter}_is_transpose"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    return indexer->ndim == 2 &&
        indexer->shape[0] >= CUMO_TRANSPOSE_TILE &&
        indexer->shape[1] >= CUMO_TRANSPOSE_TILE &&
        a1->step[1] == (ssize_t)sizeof(dtype) &&
        a1->step[0] == (ssize_t)(sizeof(dtype) * indexer->shape[1]) &&
        a2->step[0] == (ssize_t)sizeof(<%=dtype%>) &&
        a2->step[1] == (ssize_t)(sizeof(<%=dtype%>) * indexer->shape[0]);
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim, block_dim;

    if (<%="cumo_#{c_iter}_is_transpose"%>(a1, a2, indexer)) {
        uint64_t rows = indexer->shape[0];
        uint64_t cols = indexer->shape[1];
        uint64_t tiles_y = (rows + CUMO_TRANSPOSE_TILE - 1) / CUMO_TRANSPOSE_TILE;
        dim3 grid((cols + CUMO_TRANSPOSE_TILE - 1) / CUMO_TRANSPOSE_TILE,
                  (unsigned int)(tiles_y > CUMO_MAX_GRID_DIM_Y ? CUMO_MAX_GRID_DIM_Y : tiles_y));
        dim3 block(CUMO_TRANSPOSE_TILE, CUMO_TRANSPOSE_ROWS);

        <%="cumo_#{c_iter}_transpose_kernel"%><<<grid, block>>>(a1->ptr, a2->ptr, rows, cols);
        cumo_cuda_runtime_check_kernel_launch();
        return;
    }

    grid_dim = cumo_get_grid_dim(indexer->total_size);
    block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
