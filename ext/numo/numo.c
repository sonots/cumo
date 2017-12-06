#define NUMO_C
#include <ruby.h>
#include <assert.h>
#include "numo.h"

void Init_numo();
void Init_narray();
void Init_nary_data();
void Init_nary_ndloop();
void Init_nary_step();
void Init_nary_index();
void Init_numo_bit();
void Init_numo_int8();
void Init_numo_int16();
void Init_numo_int32();
void Init_numo_int64();
void Init_numo_uint8();
void Init_numo_uint16();
void Init_numo_uint32();
void Init_numo_uint64();
void Init_numo_sfloat();
void Init_numo_scomplex();
void Init_numo_dfloat();
void Init_numo_dcomplex();
void Init_numo_robject();
void Init_nary_math();
void Init_nary_rand();
void Init_nary_array();
void Init_nary_struct();
void Init_numo_cuda_driver();
void Init_numo_cuda_runtime();
void Init_numo_cuda_nvrtc();

void
numo_debug_breakpoint(void)
{
    /* */
}

/* initialization of Numo Module */
void
Init_numo()
{
    VALUE mNumo = rb_define_module("Numo");

    Init_narray();

    Init_nary_step();
    Init_nary_index();

    Init_nary_data();
    Init_nary_ndloop();

    Init_numo_dcomplex();
    Init_numo_dfloat();
    Init_numo_scomplex();
    Init_numo_sfloat();

    Init_numo_int64();
    Init_numo_uint64();
    Init_numo_int32();
    Init_numo_uint32();
    Init_numo_int16();
    Init_numo_uint16();
    Init_numo_int8();
    Init_numo_uint8();

    Init_numo_bit();
    Init_numo_robject();

    Init_nary_math();

    Init_nary_rand();
    Init_nary_array();
    Init_nary_struct();

    Init_numo_cuda_driver();
    Init_numo_cuda_runtime();
    Init_numo_cuda_nvrtc();
}
