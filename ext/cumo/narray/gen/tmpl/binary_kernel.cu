<% unless type_name == 'robject' %>

//<% if is_int and %w[div mod].include? name %>
#define cumo_check_intdivzero(y) \
    if ((y)==0) {                \
        *divzero = 1;            \
        continue;                \
    }
//<% else %>
#define cumo_check_intdivzero(y) {}
//<% end %>

//<% has_scalar = %w[add sub mul div mod bit_and bit_or bit_xor left_shift right_shift copysign].include?(name) %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
//<% if has_scalar %>
// A Ruby numeric operand rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. It reaches the
// left side through coerce, so use_scalar says which side it is on. It is the
// same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, int* divzero, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        dtype x, y;
        switch (use_scalar) {
        case 1:  x = *(dtype*)(p1); y = sv; break;
        case 2:  x = sv; y = *(dtype*)(p1); break;
        default: x = *(dtype*)(p1); y = *(dtype*)(cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer));
        }
        cumo_check_intdivzero(y);
        *(dtype*)(p3) = m_<%=name%>(x,y);
    }
}
//<% else %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, int* divzero)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        cumo_check_intdivzero(*(dtype*)(p2));
        *(dtype*)(p3) = m_<%=name%>(*(dtype*)(p1),*(dtype*)(p2));
    }
}
//<% end %>
<% end %>

// One operand of a 2-d elementwise op can be a transposed view, which leaves
// either its read or the write strided whichever way the loop walks. Stage that
// operand through a shared memory tile so all three arrays are touched in rows.
// t_side says which operand is the transposed one.
__global__ void <%="cumo_#{c_iter}_transpose_kernel"%>(char *p1, char *p2, char *p3, int* divzero, int t_side, uint64_t rows, uint64_t cols)
{
    CUMO_TRANSPOSE_TILE_DECL(dtype, tile);
    char *tp = (t_side == 1) ? p1 : p2;
    char *sp = (t_side == 1) ? p2 : p1;

    CUMO_TRANSPOSE_TILE_LOOP(tile, rows, cols,
        *(dtype*)(tp + cumo_tile_src * sizeof(dtype)),
        dtype o = *(dtype*)(sp + cumo_tile_dst * sizeof(dtype));
        dtype lhs = (t_side == 1) ? cumo_tile_val : o;
        dtype rhs = (t_side == 1) ? o : cumo_tile_val;
        cumo_check_intdivzero(rhs);
        *(dtype*)(p3 + cumo_tile_dst * sizeof(dtype)) = m_<%=name%>(lhs, rhs);
    );
}

// Launches the tiled kernel and answers 1 when it did. It takes over when one
// operand is a transposed plain array and the other two are row-major ones, and
// only once both sides reach a tile: a shorter one leaves most of each warp idle
// while the plain loop already reads or writes it in one go.
static int
<%="cumo_#{c_iter}_launch_transpose"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    ssize_t esz = (ssize_t)sizeof(dtype);
    uint64_t rows, cols;
    ssize_t row, col;
    int rows_a1, rows_a2, t_side;

    if (!CUMO_TRANSPOSE_TILE_FITS(indexer)) {
        return 0;
    }
    rows = indexer->shape[0];
    cols = indexer->shape[1];
    row = esz * (ssize_t)cols;
    col = esz * (ssize_t)rows;
    if (a3->step[0] != row || a3->step[1] != esz) {
        return 0;
    }
    rows_a1 = (a1->step[0] == row && a1->step[1] == esz);
    rows_a2 = (a2->step[0] == row && a2->step[1] == esz);
    if (!rows_a2 && rows_a1 && a2->step[0] == esz && a2->step[1] == col) {
        t_side = 2;
    } else if (!rows_a1 && rows_a2 && a1->step[0] == esz && a1->step[1] == col) {
        t_side = 1;
    } else {
        return 0;
    }

    CUMO_TRANSPOSE_LAUNCH(<%="cumo_#{c_iter}_transpose_kernel"%>, rows, cols,
                          a1->ptr, a2->ptr, a3->ptr, divzero, t_side);
    cumo_cuda_runtime_check_kernel_launch();
    return 1;
}

//<% if has_scalar %>
static void <%="cumo_#{c_iter}_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*a3,*indexer,divzero,sv,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*a3,*indexer,divzero,sv,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    dtype sv;

    if (<%="cumo_#{c_iter}_launch_transpose"%>(a1,a2,a3,indexer,divzero)) {
        return;
    }
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,a2,a3,indexer,divzero,sv,0);
}

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero, int scalar_is_left)
{
    cumo_na_iarray_t a2;
    memset(&a2, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&a2,a3,indexer,divzero,sv,scalar_is_left ? 2 : 1);
}
//<% else %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    size_t grid_dim, block_dim;

    if (<%="cumo_#{c_iter}_launch_transpose"%>(a1,a2,a3,indexer,divzero)) {
        return;
    }
    grid_dim = cumo_get_grid_dim(indexer->total_size);
    block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*a3,*indexer,divzero);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(*a1,*a2,*a3,*indexer,divzero);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
//<% end %>
#undef cumo_check_intdivzero
<% end %>
