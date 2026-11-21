<% narrow = %w[int8 uint8 int16 uint16 hfloat bfloat].include?(type_name) %>
<% if %w[dfloat dcomplex].include?(type_name) %>
// atomicAdd on double arrived with sm_60, and the default target of the CUDA
// releases cumo builds against is still below it.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ static inline void cumo_<%=type_name%>_add_at_double(double *address, double val)
{
    unsigned long long int *p = (unsigned long long int*)address;
    unsigned long long int old = *p, assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
}
#else
__device__ static inline void cumo_<%=type_name%>_add_at_double(double *address, double val) { atomicAdd(address, val); }
#endif
<% end %>
<% if narrow %>
<% bits = %w[int8 uint8].include?(type_name) ? "unsigned char" : "unsigned short" %>
// No atomic takes fewer than 32 bits on every target, so the element is added
// by swapping the aligned word around it with only its own bytes changed.
__device__ static inline void cumo_<%=type_name%>_add_at_narrow(dtype *p, dtype v)
{
    unsigned int *word = (unsigned int*)((uintptr_t)p & ~(uintptr_t)3);
    unsigned int shift = (unsigned int)((uintptr_t)p & 3) * 8;
    unsigned int mask = (unsigned int)(<%=bits%>)~0u << shift;
    unsigned int old = *word, assumed;
    do {
        <%=bits%> b = (<%=bits%>)((old & mask) >> shift);
        dtype x;
        assumed = old;
        memcpy(&x, &b, sizeof(dtype));
        x = m_add(x, v);
        memcpy(&b, &x, sizeof(dtype));
        old = atomicCAS(word, assumed, (assumed & ~mask) | ((unsigned int)b << shift));
    } while (assumed != old);
}
<% end %>

__device__ static inline void cumo_<%=type_name%>_add_at_one(dtype *p, dtype v)
{
<% case type_name
   when "int32" %>
    atomicAdd((int*)p, (int)v);
<% when "uint32" %>
    atomicAdd((unsigned int*)p, (unsigned int)v);
<% when "int64", "uint64" %>
    atomicAdd((unsigned long long int*)p, (unsigned long long int)v);
<% when "sfloat" %>
    atomicAdd(p, v);
<% when "dfloat" %>
    cumo_<%=type_name%>_add_at_double(p, v);
<% when "scomplex" %>
    atomicAdd(&p->dat[0], v.dat[0]);
    atomicAdd(&p->dat[1], v.dat[1]);
<% when "dcomplex" %>
    cumo_<%=type_name%>_add_at_double(&p->dat[0], v.dat[0]);
    cumo_<%=type_name%>_add_at_double(&p->dat[1], v.dat[1]);
<% when "hfloat" %>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    atomicAdd(p, v);
#else
    cumo_<%=type_name%>_add_at_narrow(p, v);
#endif
<% when "bfloat" %>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    atomicAdd(p, v);
#else
    cumo_<%=type_name%>_add_at_narrow(p, v);
#endif
<% else %>
    cumo_<%=type_name%>_add_at_narrow(p, v);
<% end %>
}

<% indexer_dims.each do |idim| %>
__global__ void <%="cumo_#{c_iter}_kernel_dim#{idim}"%>(cumo_na_iarray_stridx_t a1, cumo_na_iarray_stridx_t a2, cumo_na_indexer_t indexer)
{
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < indexer.total_size; i += blockDim.x * gridDim.x) {
        cumo_na_indexer_set_dim<%=idim%>(&indexer, i);
        char* p1 = cumo_na_iarray_stridx_at_dim<%=idim%>(&a1, &indexer);
        char* p2 = cumo_na_iarray_stridx_at_dim<%=idim%>(&a2, &indexer);
        cumo_<%=type_name%>_add_at_one((dtype*)p1, *(dtype*)p2);
    }
}
<% end %>

void <%="cumo_#{c_iter}_kernel_launch"%>(cumo_na_iarray_stridx_t* a1, cumo_na_iarray_stridx_t* a2, cumo_na_indexer_t* indexer)
{
    size_t grid_dim = cumo_get_grid_dim(indexer->total_size);
    size_t block_dim = cumo_get_block_dim(indexer->total_size);
    <%= indexer_switch("cumo_#{c_iter}_kernel", "*a1,*a2,*indexer") %>
    cumo_cuda_runtime_check_kernel_launch();
}
