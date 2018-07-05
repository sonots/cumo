<% unless type_name == 'robject' %>
<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

__global__ void <%="cumo_#{type_name}_#{name}#{nan}_kernel"%>(char* p1, char* p2, char* p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        *((dtype*)(p3+(i*s3))) = f_<%=name%><%=nan%>(*((dtype*)(p1+(i*s1))), *((dtype*)(p2+(i*s2))));
    }
}

void cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(char *p1, char *p2, char* p3, ssize_t s1, ssize_t s2, ssize_t s3, size_t n)
{
    size_t grid_dim = cumo_get_grid_dim(n);
    size_t block_dim = cumo_get_block_dim(n);
    <%="cumo_#{type_name}_#{name}#{nan}_kernel"%><<<grid_dim, block_dim>>>(p1,p2,p3,s1,s2,s3,n);
}

<% end %>
<% end %>
