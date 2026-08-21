<% unless type_name == 'robject' %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
// A Ruby numeric operand rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. Either side of a
// module function can be the numeric one, so use_scalar says which side sv is
// on. It is the same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        dtype x = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        dtype y;
        switch (use_scalar) {
        case 1:  y = sv; break;
        case 2:  y = x; x = sv; break;
        default: y = *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        }
        *(dtype*)cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer) = m_<%=name%>(x,y);
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

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, dtype sv, int scalar_is_left)
{
    cumo_na_iarray_t unused;
    memset(&unused, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&unused,a3,indexer,sv,scalar_is_left ? 2 : 1);
}
<% end %>
