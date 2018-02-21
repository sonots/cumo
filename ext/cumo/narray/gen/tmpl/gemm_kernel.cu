<% unless defined?($cumo_narray_gen_tmpl_gemm_kernel_included) %>
<% $cumo_narray_gen_tmpl_gemm_kernel_included = 1 %>

<% unless ['sfloat', 'dfloat', 'scomplex', 'dcomplex'].include?(type_name) %>

// TODO(sonots): Move to suitable place
#include "cublas_v2.h"

void <%="#{type_name}_gemm_kernel_launch"%>(char *p1, char *p2, char *p3, ssize_t s1, ssize_t s2, ssize_t s3, uint64_t n)
{
    size_t gridDim = get_gridDim(n);
    size_t blockDim = get_blockDim(n);

    cublasHandle_t handle;
    cublasCreate(&handle);
    dtype alpha = m_one;
    dtype beta = m_one;
    cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&alpha,a,m,b,k,&beta,c,m);
    //<%="#{type_name}_gemm#{nan}_kernel"%><<<gridDim, blockDim>>>(p1,p2,p3,s1,s2,s3,n);
    cublasDestroy(handle);
}

<% end %>
