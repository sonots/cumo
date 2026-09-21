<% unless is_object %>
__global__ void <%="cumo_#{c_func(:nodef)}_kernel"%>(dtype *ptr, dtype x)
{
    *ptr = x;
}
void <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(dtype *ptr, dtype x)
{
    <%="cumo_#{c_func(:nodef)}_kernel"%><<<1,1, 0, cumo_cuda_stream()>>>(ptr,x);
    cumo_cuda_runtime_check_kernel_launch();
}
<% end %>
