#define CUMO_C
#include <ruby.h>
#include <assert.h>
#include "cumo.h"

void Init_cumo();
void Init_cumo_narray();
void Init_cumo_nary_data();
void Init_cumo_nary_ndloop();
void Init_cumo_nary_step();
void Init_cumo_nary_index();
void Init_cumo_bit();
void Init_cumo_int8();
void Init_cumo_int16();
void Init_cumo_int32();
void Init_cumo_int64();
void Init_cumo_uint8();
void Init_cumo_uint16();
void Init_cumo_uint32();
void Init_cumo_uint64();
void Init_cumo_sfloat();
void Init_cumo_scomplex();
void Init_cumo_dfloat();
void Init_cumo_dcomplex();
void Init_cumo_robject();
void Init_cumo_nary_math();
void Init_cumo_nary_rand();
void Init_cumo_nary_array();
void Init_cumo_nary_struct();
void Init_cumo_cuda_driver();
void Init_cumo_cuda_memory_pool();
void Init_cumo_cuda_runtime();
void Init_cumo_cuda_nvrtc();

void
cumo_debug_breakpoint(void)
{
    /* */
}

/* initialization of Cumo Module */
void
Init_cumo()
{
    rb_define_module("Cumo");

    Init_cumo_narray();

    Init_cumo_nary_step();
    Init_cumo_nary_index();

    Init_cumo_nary_data();
    Init_cumo_nary_ndloop();

    Init_cumo_dcomplex();
    Init_cumo_dfloat();
    Init_cumo_scomplex();
    Init_cumo_sfloat();

    Init_cumo_int64();
    Init_cumo_uint64();
    Init_cumo_int32();
    Init_cumo_uint32();
    Init_cumo_int16();
    Init_cumo_uint16();
    Init_cumo_int8();
    Init_cumo_uint8();

    Init_cumo_bit();
    Init_cumo_robject();

    Init_cumo_nary_math();

    Init_cumo_nary_rand();
    Init_cumo_nary_array();
    Init_cumo_nary_struct();

    Init_cumo_cuda_driver();
    Init_cumo_cuda_memory_pool();
    Init_cumo_cuda_runtime();
    Init_cumo_cuda_nvrtc();
}
