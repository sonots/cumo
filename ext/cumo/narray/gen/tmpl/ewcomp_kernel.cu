<% unless type_name == 'robject' %>
<% (is_float ? ["","_nan"] : [""]).each do |nan| %>

__global__ void <%="cumo_#{type_name}_#{name}#{nan}_kernel"%>(dtype* p1, dtype* p2, dtype* p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        p3[i*s3] = f_<%=name%><%=nan%>(p1[i*s1], p2[i*s2]);
    }
}

void cumo_<%=type_name%>_<%=name%><%=nan%>_kernel_launch(char *p1, ssize_t s1, char *p2, ssize_t s2, char* p3, ssize_t s3, size_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);
    <%="cumo_#{type_name}_#{name}#{nan}_kernel"%><<<gridDim, blockDim>>>((dtype*)p1,(dtype*)p2,(dtype*)p3,s1/sizeof(dtype),s2/sizeof(dtype),s3/sizeof(dtype),n);
}

<% end %>
<% end %>
