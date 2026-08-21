<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
// A Ruby numeric bound rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. use_scalar is the
// same for every thread, so the branches cost nothing, and it also says the
// caller already compared the two bounds, so invalid is never written.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_iarray_t a4, cumo_na_indexer_t indexer, int* invalid, dtype sv_min, dtype sv_max, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype min = use_scalar ? sv_min : *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        dtype max = use_scalar ? sv_max : *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        // Leaving the element unwritten keeps a scalar min > max, where every
        // element takes this branch, from touching the output before it raises.
        if (!use_scalar && m_gt(min,max)) { *invalid = 1; continue; }
        if (m_lt(x,min)) { x = min; }
        if (m_gt(x,max)) { x = max; }
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a4, &indexer) = x;
    }
}

__global__ void <%="cumo_#{c_iter}_min_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype min = use_scalar ? sv : *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer) = m_lt(x,min) ? min : x;
    }
}

__global__ void <%="cumo_#{c_iter}_max_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype max = use_scalar ? sv : *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer) = m_gt(x,max) ? max : x;
    }
}
<% end %>

static void <%="cumo_#{c_iter}_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer, int* invalid, dtype sv_min, dtype sv_max, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*a4,*indexer,invalid,sv_min,sv_max,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*a4,*indexer,invalid,sv_min,sv_max,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer, int* invalid)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,a2,a3,a4,indexer,invalid,sv,sv,0);
}

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv_min, dtype sv_max, cumo_na_iarray_t* a4, cumo_na_indexer_t* indexer)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&unused,&unused,a4,indexer,0,sv_min,sv_max,1);
}

static void <%="cumo_#{c_iter}_min_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_min_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_min_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_min_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_min_kernel_dispatch"%>(a1,a2,a3,indexer,sv,0);
}

void <%="cumo_#{c_iter}_min_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_min_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,1);
}

static void <%="cumo_#{c_iter}_max_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_max_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_max_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,sv,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_max_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_max_kernel_dispatch"%>(a1,a2,a3,indexer,sv,0);
}

void <%="cumo_#{c_iter}_max_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_max_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,1);
}
<% end %>
