void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_reduction_arg_t* arg)
{
    cumo_bit_stat_reduce(*arg, <%=stat%>);
}
