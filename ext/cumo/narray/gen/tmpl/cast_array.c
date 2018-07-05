static VALUE
<%=c_func(:nodef)%>(VALUE rary)
{
    VALUE nary;
    cumo_narray_t *na;

    nary = cumo_na_s_new_like(cT, rary);
    CumoGetNArray(nary,na);
    if (na->size > 0) {
        <%=find_tmpl("store").find("array").c_func%>(nary,rary);
    }
    return nary;
}
