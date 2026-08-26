void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_bit_reduction_arg_t* arg)
{
    // m_<%=name%>(0) is 1 exactly when the zeros are what this method counts.
    cumo_bit_count_reduce(*arg, m_<%=name%>(0));
}
