<% if type_name == 'robject' %>
<% else %>

//<% if is_int and !is_unsigned and name == 'abs' %>
#define cumo_check_intmin(x) \
    if ((x)==DATA_MIN) {     \
        *intmin = 1;         \
        continue;            \
    }
//<% else %>
#define cumo_check_intmin(x) {}
//<% end %>

<% ((0..opt_indexer_ndim).to_a << '').each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_t a1, cumo_na_iarray_t a2, cumo_na_indexer_t indexer, int* intmin)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_at_dim<%=idim%>(&a2, &indexer);
        cumo_check_intmin(*(dtype*)(p1));
        *(<%=dtype%>*)(p2) = m_<%=name%>(*(dtype*)(p1));
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_t* a1, cumo_na_iarray_t* a2, cumo_na_indexer_t* indexer, int* intmin)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    switch (indexer->ndim) {
    <% (0..opt_indexer_ndim).each do |idim| %>
    case <%=idim%>:
        <%="cumo_#{c_iter}_kernel_dim#{idim}"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer,intmin);
        break;
    <% end %>
    default:
        <%="cumo_#{c_iter}_kernel_dim"%><<<grid_dim, block_dim>>>(*a1,*a2,*indexer,intmin);
        break;
    }
    cumo_cuda_runtime_check_kernel_launch();
}
#undef cumo_check_intmin
<% end %>
