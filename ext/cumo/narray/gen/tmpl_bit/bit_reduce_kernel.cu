void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_pred_reduction_arg_t* arg)
{
    // init_bit is 1 for all? and 0 for any?, which is also what separates the
    // two answers a reduced count can give.
    cumo_bit_pred_reduce(*arg, <%=init_bit%>);
}
