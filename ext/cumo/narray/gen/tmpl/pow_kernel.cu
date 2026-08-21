<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
// A Ruby numeric exponent rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. use_scalar is the
// same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        dtype v = *(dtype*)(p1);
        dtype x, y;
        switch (use_scalar) {
        case 1:  x = v; y = sv; break;
        case 2:  x = sv; y = v; break;
        default: x = v; y = *(dtype*)(cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer));
        }
        *(dtype*)(p3) = m_pow(x, y);
    }
}

__global__ void <%="cumo_#{c_iter}_int32_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, int32_t sv, int use_scalar, dtype sv_base)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        int32_t y = use_scalar == 1 ? sv : *(int32_t*)(cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer));
        *(dtype*)(p3) = use_scalar == 2 ? m_pow_int(sv_base, y) : m_pow_int(*(dtype*)(p1), y);
    }
}
<% end %>

static void <%="cumo_#{c_iter}_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,a2,a3,indexer,sv,0);
}

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int scalar_is_left)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,scalar_is_left ? 2 : 1);
}

static void <%="cumo_#{c_iter}_int32_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int32_t sv, int use_scalar, dtype sv_base)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_int32_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar,sv_base);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_int32_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar,sv_base);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_int32_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    dtype sv_base;
    memset(&sv_base, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_int32_kernel_dispatch"%>(a1,a2,a3,indexer,0,0,sv_base);
}

void <%="cumo_#{c_iter}_int32_s_kernel_launch"%>(cumo_na_iarray_t* a1, int32_t sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    cumo_na_iarray_t unused;
    dtype sv_base;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    memset(&sv_base, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_int32_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,1,sv_base);
}

// The base is the numeric one when it came in through coerce, and the exponent
// is the Int32 array.
void <%="cumo_#{c_iter}_int32_s_left_kernel_launch"%>(dtype sv_base, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_int32_kernel_dispatch"%>(&unused,a2,a3,indexer,0,2,sv_base);
}
<% end %>
