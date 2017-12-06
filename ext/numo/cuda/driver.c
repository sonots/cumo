#include <ruby.h>
#include <ruby/thread.h>
#include <cuda.h>
#include "numo/cuda/driver.h"

VALUE numo_cuda_eDriverError;
VALUE numo_cuda_mDriver;
#define eDriverError numo_cuda_eDriverError
#define mDriver numo_cuda_mDriver

static void
check_status(CUresult status)
{
    if (status != 0) {
        const char *errname = NULL;
        const char *errstring = NULL;
        cuGetErrorString(status, &errname);
        cuGetErrorString(status, &errstring);
        rb_raise(numo_cuda_eDriverError, "%s %s", errname, errstring);
    }
}

///////////////////////////////////////////////
// Context Management
//////////////////////////////////////////////

static VALUE
rb_cuCtxGetCurrent(VALUE self)
{
    CUcontext ctx;
    CUresult status;

    status = cuCtxGetCurrent(&ctx);
    check_status(status);

    return SIZET2NUM((size_t)ctx);
}

///////////////////////////////////////////////
// Module Load and Kernel Execution
//////////////////////////////////////////////

struct cuLinkAddDataParam {
    CUlinkState state;
    CUjitInputType type;
    void* data;
    size_t size;
    const char* name;
    unsigned int numOptions;
    CUjit_option* options;
    void ** optionValues;
};

static void *
cuLinkAddData_without_gvl_cb(void *param)
{
    struct cuLinkAddDataParam *p = param;
    CUresult status;
    status = cuLinkAddData(p->state, p->type, p->data, p->size, p->name, p->numOptions, p->options, p->optionValues);
    return (void *)status;
}

// TODO(sonots): Support options.
static VALUE
rb_cuLinkAddData(VALUE self, VALUE state, VALUE type, VALUE data, VALUE name)
{
    CUlinkState _state = (CUlinkState)NUM2SIZET(state);
    CUjitInputType _type = (CUjitInputType)NUM2INT(type);
    void* _data = (void *)RSTRING_PTR(data);
    size_t _size = RSTRING_LEN(data);
    const char* _name = RSTRING_PTR(data);
    CUresult status;

    struct cuLinkAddDataParam param = {_state, _type, _data, _size, _name, 0, (CUjit_option*)0, (void**)0};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkAddData_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return Qnil;
}

struct cuLinkAddFileParam {
    CUlinkState state;
    CUjitInputType type;
    const char* path;
    unsigned int numOptions;
    CUjit_option* options;
    void ** optionValues;
};

static void *
cuLinkAddFile_without_gvl_cb(void *param)
{
    struct cuLinkAddFileParam *p = param;
    CUresult status;
    status = cuLinkAddFile(p->state, p->type, p->path, p->numOptions, p->options, p->optionValues);
    return (void *)status;
}

// TODO(sonots): Support options.
static VALUE
rb_cuLinkAddFile(VALUE self, VALUE state, VALUE type, VALUE path)
{
    CUlinkState _state = (CUlinkState)NUM2SIZET(state);
    CUjitInputType _type = (CUjitInputType)NUM2INT(type);
    const char* _path = RSTRING_PTR(path);
    CUresult status;

    struct cuLinkAddFileParam param = {_state, _type, _path, 0, (CUjit_option*)0, (void **)0};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkAddFile_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return Qnil;
}

struct cuLinkCompleteParam {
    CUlinkState state;
    void** cubinOut;
    size_t* sizeOut;
};

static void *
cuLinkComplete_without_gvl_cb(void *param)
{
    struct cuLinkCompleteParam *p = param;
    CUresult status;
    status = cuLinkComplete(p->state, p->cubinOut, p->sizeOut);
    return (void *)status;
}

static VALUE
rb_cuLinkComplete(VALUE self, VALUE state)
{
    CUlinkState _state = (CUlinkState)NUM2SIZET(state);
    void* _cubinOut;
    size_t _sizeOut;
    CUresult status;

    struct cuLinkCompleteParam param = {_state, &_cubinOut, &_sizeOut};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkComplete_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return rb_str_new((char *)_cubinOut, _sizeOut);
}

struct cuLinkCreateParam {
    unsigned int numOptions;
    CUjit_option* options;
    void** optionValues;
    CUlinkState* state;
};

static void *
cuLinkCreate_without_gvl_cb(void *param)
{
    struct cuLinkCreateParam *p = param;
    CUresult status;
    status = cuLinkCreate(p->numOptions, p->options, p->optionValues, p->state);
    return (void *)status;
}

// TODO(sonots): Support options.
static VALUE
rb_cuLinkCreate(VALUE self)
{
    CUlinkState state;
    CUresult status;

    struct cuLinkCreateParam param = {0, (CUjit_option*)0, (void**)0, &state};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkCreate_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return SIZET2NUM((size_t)state);
}

struct cuLinkDestroyParam {
    CUlinkState state;
};

static void *
cuLinkDestroy_without_gvl_cb(void *param)
{
    struct cuLinkDestroyParam *p = param;
    CUresult status;
    status = cuLinkDestroy(p->state);
    return (void *)status;
}

static VALUE
rb_cuLinkDestroy(VALUE self, VALUE state)
{
    CUlinkState _state = (CUlinkState)NUM2SIZET(state);
    CUresult status;

    struct cuLinkDestroyParam param = {_state};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkDestroy_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return Qnil;
}

struct cuModuleGetFunctionParam {
    CUfunction* hfunc;
    CUmodule hmod;
    const char* name;
};

static void *
cuModuleGetFunction_without_gvl_cb(void *param)
{
    struct cuModuleGetFunctionParam *p = param;
    CUresult status;
    status = cuModuleGetFunction(p->hfunc, p->hmod, p->name);
    return (void *)status;
}

static VALUE
rb_cuModuleGetFunction(VALUE self, VALUE hmod, VALUE name)
{
    CUfunction _hfunc;
    CUmodule _hmod = (CUmodule)NUM2SIZET(hmod);
    const char* _name = RSTRING_PTR(name);
    CUresult status;

    struct cuModuleGetFunctionParam param = {&_hfunc, _hmod, _name};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleGetFunction_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return SIZET2NUM((size_t)_hfunc);
}

struct cuModuleGetGlobalParam {
    CUdeviceptr* dptr;
    size_t* bytes;
    CUmodule hmod;
    const char* name;
};

static void *
cuModuleGetGlobal_without_gvl_cb(void *param)
{
    struct cuModuleGetGlobalParam *p = param;
    CUresult status;
    status = cuModuleGetGlobal(p->dptr, p->bytes, p->hmod, p->name);
    return (void *)status;
}

static VALUE
rb_cuModuleGetGlobal(VALUE self, VALUE hmod, VALUE name)
{
    CUdeviceptr _dptr;
    size_t _bytes;
    CUmodule _hmod = (CUmodule)NUM2SIZET(hmod);
    const char* _name = RSTRING_PTR(name);
    CUresult status;

    struct cuModuleGetGlobalParam param = {&_dptr, &_bytes, _hmod, _name};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleGetGlobal_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return rb_str_new((char *)_dptr, _bytes);
}

struct cuModuleLoadParam {
    CUmodule* module;
    const char* fname;
};

static void *
cuModuleLoad_without_gvl_cb(void *param)
{
    struct cuModuleLoadParam *p = param;
    CUresult status;
    status = cuModuleLoad(p->module, p->fname);
    return (void *)status;
}

static VALUE
rb_cuModuleLoad(VALUE self, VALUE fname)
{
    CUmodule _module;
    const char* _fname = RSTRING_PTR(fname);
    CUresult status;

    struct cuModuleLoadParam param = {&_module, _fname};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleLoad_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return SIZET2NUM((size_t)_module);
}

struct cuModuleLoadDataParam {
    CUmodule* module;
    const void* image;
};

static void *
cuModuleLoadData_without_gvl_cb(void *param)
{
    struct cuModuleLoadDataParam *p = param;
    CUresult status;
    status = cuModuleLoadData(p->module, p->image);
    return (void *)status;
}

static VALUE
rb_cuModuleLoadData(VALUE self, VALUE image)
{
    CUmodule _module;
    const void* _image = (void*)NUM2SIZET(image);
    CUresult status;

    struct cuModuleLoadDataParam param = {&_module, _image};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleLoadData_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return SIZET2NUM((size_t)_module);
}

struct cuModuleUnloadParam {
    CUmodule hmod;
};

static void *
cuModuleUnload_without_gvl_cb(void *param)
{
    struct cuModuleUnloadParam *p = param;
    CUresult status;
    status = cuModuleUnload(p->hmod);
    return (void *)status;
}

static VALUE
rb_cuModuleUnload(VALUE self, VALUE hmod)
{
    CUmodule _hmod = (CUmodule)NUM2SIZET(hmod);
    CUresult status;

    struct cuModuleUnloadParam param = {_hmod};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleUnload_without_gvl_cb, &param, NULL, NULL);

    check_status(status);
    return Qnil;
}

void
Init_numo_cuda_driver()
{
    VALUE mNumo = rb_define_module("Numo");
    VALUE mCUDA = rb_define_module_under(mNumo, "CUDA");
    mDriver = rb_define_module_under(mCUDA, "Driver");
    eDriverError = rb_define_class_under(mCUDA, "DriverError", rb_eStandardError);

    rb_define_singleton_method(mDriver, "cuCtxGetCurrent", rb_cuCtxGetCurrent, 0);
    rb_define_singleton_method(mDriver, "cuLinkAddData",   rb_cuLinkAddData,   4);
    rb_define_singleton_method(mDriver, "cuLinkAddFile", rb_cuLinkAddFile, 3);
    rb_define_singleton_method(mDriver, "cuLinkComplete", rb_cuLinkComplete, 1);
    rb_define_singleton_method(mDriver, "cuLinkCreate", rb_cuLinkCreate, 0);
    rb_define_singleton_method(mDriver, "cuLinkDestroy", rb_cuLinkDestroy, 1);
    rb_define_singleton_method(mDriver, "cuModuleGetFunction", rb_cuModuleGetFunction, 2);
    rb_define_singleton_method(mDriver, "cuModuleGetGlobal", rb_cuModuleGetGlobal, 2);
    rb_define_singleton_method(mDriver, "cuModuleLoad", rb_cuModuleLoad, 1);
    rb_define_singleton_method(mDriver, "cuModuleLoadData", rb_cuModuleLoadData, 1);
    rb_define_singleton_method(mDriver, "cuModuleUnload", rb_cuModuleUnload, 1);

    rb_define_const(mDriver, "CU_JIT_INPUT_CUBIN", INT2NUM(CU_JIT_INPUT_CUBIN));
    rb_define_const(mDriver, "CU_JIT_INPUT_FATBINARY", INT2NUM(CU_JIT_INPUT_FATBINARY));
    rb_define_const(mDriver, "CU_JIT_INPUT_LIBRARY", INT2NUM(CU_JIT_INPUT_LIBRARY));
    rb_define_const(mDriver, "CU_JIT_INPUT_OBJECT", INT2NUM(CU_JIT_INPUT_OBJECT));
    rb_define_const(mDriver, "CU_JIT_INPUT_PTX", INT2NUM(CU_JIT_INPUT_PTX));
}
