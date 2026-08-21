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

//<% has_scalar = %w[add sub mul div].include?(name) %>
<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
//<% if has_scalar %>
// A Ruby numeric operand rides in as sv rather than as a 0-dimensional array,
// which would cost a whole kernel launch of its own to fill. use_scalar is the
// same for every thread, so the branch costs nothing.
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_iarray_t a3, cumo_na_indexer_t indexer, int* divzero, dtype sv, int use_scalar)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p3 = cumo_na_iarray_at_dim<%=idim%>(&a3, &indexer);
        dtype y = use_scalar ? sv : *(dtype*)(cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer));
        cumo_check_intdivzero(y);
        *(dtype*)(p3) = m_<%=name%>(*(dtype*)(p1),y);
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

//<% if has_scalar %>
static void <%="cumo_#{c_iter}_kernel_dispatch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero, dtype sv, int use_scalar)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,divzero,sv,use_scalar);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,divzero,sv,use_scalar);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    dtype sv;
    memset(&sv, 0, sizeof(dtype));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,a2,a3,indexer,divzero,sv,0);
}

void <%="cumo_#{c_iter}_s_kernel_launch"%>(cumo_na_iarray_t* a1, dtype sv, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    cumo_na_iarray_t a2;
    memset(&a2, 0, sizeof(cumo_na_iarray_t));
    <%="cumo_#{c_iter}_kernel_dispatch"%>(a1,&a2,a3,indexer,divzero,sv,1);
}
//<% else %>
void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_iarray_t* a3, cumo_na_indexer_t* indexer, int* divzero)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,divzero);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*a3,*indexer,divzero);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
//<% end %>
#undef cumo_check_intdivzero
<% end %>
