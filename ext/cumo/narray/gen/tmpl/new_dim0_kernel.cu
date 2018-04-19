__global__ void <%="cumo_#{c_func(:nodef)}_kernel"%>(dtype *ptr, dtype x)
{
    *ptr = x;
}
void <%="cumo_#{c_func(:nodef)}_kernel_launch"%>(dtype *ptr, dtype x)
{
    <%="cumo_#{c_func(:nodef)}_kernel"%><<<1,1>>>(ptr,x);
}
