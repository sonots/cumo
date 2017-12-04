#include <ruby.h>
#include <assert.h>
#include <nvrtc.h>
//#include "nomo/cuda/nvrtc.h"

VALUE cumo_cuda_eNVRTCError;
VALUE cumo_cuda_mNVRTC;
#define eNVRTCError cumo_cuda_eNVRTCError
#define mNVRTC cumo_cuda_mNVRTC

static void
check_status(int status)
{
    if (status != 0) {
        rb_raise(cumo_cuda_eNVRTCError, "NVRTC error: %d %s", status, nvrtcGetErrorString(status));
    }
}

static VALUE
compile_program(VALUE self, VALUE prog, VALUE options)
{
    int i;
    VALUE option;
    int status;
    void * _prog = (void *)NUM2SIZET(prog);
    int option_num = RARRAY_LEN(options);
    const char** option_vec = (const char **)malloc(option_num);
    for (i = 0; i < option_num; i++) {
        option = RARRAY_PTR(options)[i];
        option_vec[i] = StringValueCStr(option);
    }
    status = nvrtcCompileProgram(_prog, option_num, option_vec);
    free(option_vec);
    check_status(status);
    return Qnil;
}

void
Init_cumo_narray_cuda_nvrtc()
{
    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    mNVRTC = rb_define_module_under(mCUDA, "NVRTC");
    eNVRTCError = rb_define_class_under(mCUDA, "NVRTCError", rb_eStandardError);

    //rb_define_singleton_method(cNVRTC, "create_program", create_program, -1);
    //rb_define_singleton_method(cNVRTC, "destroy_program", destroy_program, -1);
    rb_define_singleton_method(mNVRTC, "compile_program", compile_program, 2);
    //rb_define_singleton_method(cNVRTC, "get_ptx", get_ptx, -1);
    //rb_define_singleton_method(cNVRTC, "get_program_log", get_program_log, -1);
}
