#include <ruby.h>
#include <ruby/thread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cumo/cuda/driver.h"
#include "cumo/cuda/handle.h"
#include "cumo/narray.h"
#include "cumo/intern.h"

VALUE cumo_cuda_eDriverError;
VALUE cumo_cuda_mDriver;
#define eDriverError cumo_cuda_eDriverError
#define mDriver cumo_cuda_mDriver

static cumo_cuda_handle_set_t link_states;
static cumo_cuda_handle_set_t modules;
// A function is only as live as the module it came from, so the table keeps
// the owning module against each CUfunction.
static cumo_cuda_handle_set_t functions;

static void
check_status(CUresult status)
{
    if (status != 0) {
        const char *errname = NULL;
        const char *errstring = NULL;
        cuGetErrorName(status, &errname);
        cuGetErrorString(status, &errstring);
        rb_raise(cumo_cuda_eDriverError, "%s %s (error=%d)", errname, errstring, status);
    }
}

///////////////////////////////////////////////
// Context Management
//////////////////////////////////////////////

static VALUE
rb_cuCtxCreate(VALUE self, VALUE flags, VALUE dev)
{
    unsigned int _flags = NUM2INT(flags);
    CUdevice _dev = (CUdevice)NUM2INT(dev);
    CUcontext _pctx;
    CUresult status;

#if defined(CUDA_VERSION) && CUDA_VERSION >= 13000
    status = cuCtxCreate(&_pctx, NULL, _flags, _dev);
#else
    status = cuCtxCreate(&_pctx, _flags, _dev);
#endif

    check_status(status);
    return SIZET2NUM((size_t)_pctx);
}

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
// Device Management
//////////////////////////////////////////////

static VALUE
rb_cuDeviceGet(VALUE self, VALUE ordinal)
{
    int _ordinal = NUM2INT(ordinal);
    CUdevice _device;
    CUresult status;

    status = cuDeviceGet(&_device, _ordinal);

    check_status(status);
    return INT2NUM(_device);
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
    CUjitInputType _type = (CUjitInputType)NUM2INT(type);
    // The image may be a cubin, so it is taken by length and allowed to hold
    // NUL bytes; the name is a plain C string cuLinkAddData reports errors with.
    void* _data = (void *)StringValuePtr(data);
    size_t _size = RSTRING_LEN(data);
    const char* _name = StringValueCStr(name);
    CUlinkState _state = (CUlinkState)cumo_cuda_handle_get(&link_states, state, "CUlinkState");
    CUresult status;

    struct cuLinkAddDataParam param = {_state, _type, _data, _size, _name, 0, (CUjit_option*)0, (void**)0};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkAddData_without_gvl_cb, &param, NULL, NULL);
    //status = cuLinkAddData(_state, _type, _data, _size, _name, 0, (CUjit_option*)0, (void**)0);

    RB_GC_GUARD(data);
    RB_GC_GUARD(name);
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
    CUjitInputType _type = (CUjitInputType)NUM2INT(type);
    const char* _path = StringValueCStr(path);
    CUlinkState _state = (CUlinkState)cumo_cuda_handle_get(&link_states, state, "CUlinkState");
    CUresult status;

    struct cuLinkAddFileParam param = {_state, _type, _path, 0, (CUjit_option*)0, (void **)0};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkAddFile_without_gvl_cb, &param, NULL, NULL);
    //status = cuLinkAddFile(_state, _type, _path, 0, (CUjit_option*)0, (void **)0);

    RB_GC_GUARD(path);
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
    CUlinkState _state = (CUlinkState)cumo_cuda_handle_get(&link_states, state, "CUlinkState");
    void* _cubinOut;
    size_t _sizeOut;
    CUresult status;

    struct cuLinkCompleteParam param = {_state, &_cubinOut, &_sizeOut};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkComplete_without_gvl_cb, &param, NULL, NULL);
    //status = cuLinkComplete(_state, &_cubinOut, &_sizeOut);

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
    //status = cuLinkCreate(0, (CUjit_option*)0, (void**)0, &state);

    check_status(status);
    cumo_cuda_handle_set_add(&link_states, (size_t)state);
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
    CUlinkState _state = (CUlinkState)cumo_cuda_handle_take(&link_states, state, "CUlinkState");
    CUresult status;

    struct cuLinkDestroyParam param = {_state};
    status = (CUresult)rb_thread_call_without_gvl(cuLinkDestroy_without_gvl_cb, &param, NULL, NULL);
    //status = cuLinkDestroy(_state);

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
    const char* _name = StringValueCStr(name);
    CUmodule _hmod = (CUmodule)cumo_cuda_handle_get(&modules, hmod, "CUmodule");
    CUresult status;

    struct cuModuleGetFunctionParam param = {&_hfunc, _hmod, _name};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleGetFunction_without_gvl_cb, &param, NULL, NULL);
    //status = cuModuleGetFunction(&_hfunc, _hmod, _name);

    RB_GC_GUARD(name);
    check_status(status);

    rb_nativethread_lock_lock(&functions.lock);
    st_insert(functions.table, (st_data_t)_hfunc, (st_data_t)_hmod);
    rb_nativethread_lock_unlock(&functions.lock);
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
    const char* _name = StringValueCStr(name);
    CUmodule _hmod = (CUmodule)cumo_cuda_handle_get(&modules, hmod, "CUmodule");
    CUresult status;
    VALUE ret;

    struct cuModuleGetGlobalParam param = {&_dptr, &_bytes, _hmod, _name};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleGetGlobal_without_gvl_cb, &param, NULL, NULL);
    //status = cuModuleGetGlobal(&_dptr, &_bytes, _hmod, _name);

    RB_GC_GUARD(name);
    check_status(status);

    // _dptr addresses device memory, which the host cannot read directly.
    ret = rb_str_new(NULL, (long)_bytes);
    check_status(cuMemcpyDtoH(RSTRING_PTR(ret), _dptr, _bytes));
    return ret;
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
    const char* _fname = StringValueCStr(fname);
    CUresult status;

    struct cuModuleLoadParam param = {&_module, _fname};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleLoad_without_gvl_cb, &param, NULL, NULL);
    //status = cuModuleLoad(&_module, _fname);

    RB_GC_GUARD(fname);
    check_status(status);
    cumo_cuda_handle_set_add(&modules, (size_t)_module);
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
    // A cubin is binary, so the image is not required to be NUL-free.
    const void* _image = (void*)StringValuePtr(image);
    CUresult status;

    struct cuModuleLoadDataParam param = {&_module, _image};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleLoadData_without_gvl_cb, &param, NULL, NULL);
    //status = cuModuleLoadData(&_module, _image);

    RB_GC_GUARD(image);
    check_status(status);
    cumo_cuda_handle_set_add(&modules, (size_t)_module);
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

static int
forget_function(st_data_t key, st_data_t owner, st_data_t hmod)
{
    return owner == hmod ? ST_DELETE : ST_CONTINUE;
}

static VALUE
rb_cuModuleUnload(VALUE self, VALUE hmod)
{
    CUmodule _hmod = (CUmodule)cumo_cuda_handle_take(&modules, hmod, "CUmodule");
    CUresult status;

    // The driver hands the address out again for the next module, so a
    // function of this one must not pass as live once that happens.
    rb_nativethread_lock_lock(&functions.lock);
    st_foreach(functions.table, forget_function, (st_data_t)_hmod);
    rb_nativethread_lock_unlock(&functions.lock);

    struct cuModuleUnloadParam param = {_hmod};
    status = (CUresult)rb_thread_call_without_gvl(cuModuleUnload_without_gvl_cb, &param, NULL, NULL);
    //status = cuModuleUnload(_hmod);

    check_status(status);
    return Qnil;
}

///////////////////////////////////////////////
// Execution Control
//////////////////////////////////////////////

static CUfunction
function_get(VALUE v)
{
    size_t handle = NUM2SIZET(v);
    int found;

    rb_nativethread_lock_lock(&functions.lock);
    found = st_lookup(functions.table, (st_data_t)handle, 0);
    rb_nativethread_lock_unlock(&functions.lock);
    if (!found) {
        rb_raise(rb_eArgError, "not a live CUfunction");
    }
    return (CUfunction)handle;
}

typedef union {
    void *ptr;
    long long ll;
    double d;
} kernel_arg_t;

static int
kernel_arg_is_narray(VALUE v)
{
    if (!rb_obj_is_kind_of(v, cumo_cNArray)) return 0;
    if (rb_obj_is_kind_of(v, cumo_cBit) || rb_obj_is_kind_of(v, cumo_cRObject)) {
        rb_raise(rb_eTypeError, "a %s cannot be handed to a kernel", rb_obj_classname(v));
    }
    if (cumo_na_check_contiguous(v) != Qtrue) {
        rb_raise(rb_eArgError, "a kernel takes a contiguous NArray, and this one is a view with a stride or an index");
    }
    return 1;
}

static size_t
kernel_arg_set(VALUE v, kernel_arg_t *slot, void **param)
{
    if (RB_TYPE_P(v, T_STRING)) {
        *param = RSTRING_PTR(v);
        return (size_t)RSTRING_LEN(v);
    } else if (RB_FLOAT_TYPE_P(v)) {
        slot->d = NUM2DBL(v);
        *param = &slot->d;
        return sizeof(double);
    } else if (RB_INTEGER_TYPE_P(v)) {
        slot->ll = NUM2LL(v);
        *param = &slot->ll;
        return sizeof(long long);
    }
    rb_raise(rb_eTypeError, "a kernel argument is an NArray, an Integer, a Float or a String of packed bytes, not a %s", rb_obj_classname(v));
}

#if CUDA_VERSION >= 12040
// The driver knows the kernel's parameters from 12.4 on, so a count or a
// size that does not match is refused here rather than read past.
static void
check_params(CUfunction f, long n, const size_t *given)
{
    size_t offset, size;
    long i;

    for (i = 0; i < n; i++) {
        if (cuFuncGetParamInfo(f, (size_t)i, &offset, &size) != CUDA_SUCCESS) {
            rb_raise(rb_eArgError, "the kernel takes %ld arguments, %ld were given", i, n);
        }
        if (size != given[i]) {
            rb_raise(rb_eArgError, "argument %ld is %"PRIuSIZE" bytes where the kernel takes %"PRIuSIZE, i, given[i], size);
        }
    }
    if (cuFuncGetParamInfo(f, (size_t)n, &offset, &size) == CUDA_SUCCESS) {
        rb_raise(rb_eArgError, "the kernel takes more than %ld arguments", n);
    }
}
#endif

static VALUE
rb_cuLaunchKernel(VALUE self, VALUE hfunc,
                  VALUE grid_x, VALUE grid_y, VALUE grid_z,
                  VALUE block_x, VALUE block_y, VALUE block_z,
                  VALUE shared_mem, VALUE stream, VALUE args)
{
    CUfunction f = function_get(hfunc);
    long i, n;
    VALUE slots_buf, params_buf, sizes_buf;
    kernel_arg_t *slots;
    void **params;
    size_t *sizes;
    CUresult status;

    if (NUM2SIZET(stream) != 0) {
        rb_raise(rb_eArgError, "a stream other than 0 is not supported yet");
    }
    Check_Type(args, T_ARRAY);
    n = RARRAY_LEN(args);
    slots = ALLOCV_N(kernel_arg_t, slots_buf, n);
    params = ALLOCV_N(void*, params_buf, n);
    sizes = ALLOCV_N(size_t, sizes_buf, n);
    // Taking an NArray's pointer can allocate it, which runs Ruby code and
    // with it the GC, so every NArray is resolved before a String's pointer
    // is taken.
    for (i = 0; i < n; i++) {
        VALUE v = RARRAY_AREF(args, i);
        if (kernel_arg_is_narray(v)) {
            slots[i].ptr = cumo_na_get_offset_pointer_for_write(v);
            params[i] = &slots[i].ptr;
            sizes[i] = sizeof(void*);
        } else {
            params[i] = NULL;
        }
    }
    for (i = 0; i < n; i++) {
        if (params[i] == NULL) {
            sizes[i] = kernel_arg_set(RARRAY_AREF(args, i), &slots[i], &params[i]);
        }
    }
#if CUDA_VERSION >= 12040
    check_params(f, n, sizes);
#endif
    status = cuLaunchKernel(f,
                            NUM2UINT(grid_x), NUM2UINT(grid_y), NUM2UINT(grid_z),
                            NUM2UINT(block_x), NUM2UINT(block_y), NUM2UINT(block_z),
                            NUM2UINT(shared_mem), (CUstream)0,
                            params, NULL);
    ALLOCV_END(sizes_buf);
    ALLOCV_END(params_buf);
    ALLOCV_END(slots_buf);
    RB_GC_GUARD(args);
    check_status(status);
    return Qnil;
}

void
Init_cumo_cuda_driver()
{
    CUdevice cuDevice;
    CUcontext context;

    VALUE mCumo = rb_define_module("Cumo");
    VALUE mCUDA = rb_define_module_under(mCumo, "CUDA");
    mDriver = rb_define_module_under(mCUDA, "Driver");
    eDriverError = rb_define_class_under(mCUDA, "DriverError", rb_eStandardError);

    cumo_cuda_handle_set_init(&link_states);
    cumo_cuda_handle_set_init(&modules);
    cumo_cuda_handle_set_init(&functions);

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
    rb_define_singleton_method(mDriver, "cuLaunchKernel", rb_cuLaunchKernel, 10);

    rb_define_singleton_method(mDriver, "cuDeviceGet", rb_cuDeviceGet, 1);
    rb_define_singleton_method(mDriver, "cuCtxCreate", rb_cuCtxCreate, 2);

    rb_define_const(mDriver, "CU_JIT_INPUT_CUBIN", INT2NUM(CU_JIT_INPUT_CUBIN));
    rb_define_const(mDriver, "CU_JIT_INPUT_FATBINARY", INT2NUM(CU_JIT_INPUT_FATBINARY));
    rb_define_const(mDriver, "CU_JIT_INPUT_LIBRARY", INT2NUM(CU_JIT_INPUT_LIBRARY));
    rb_define_const(mDriver, "CU_JIT_INPUT_OBJECT", INT2NUM(CU_JIT_INPUT_OBJECT));
    rb_define_const(mDriver, "CU_JIT_INPUT_PTX", INT2NUM(CU_JIT_INPUT_PTX));

    check_status(cuInit(0));

    // A driver API call needs a current context, and the runtime API only
    // creates its primary one once an array operation happens, so this covers
    // the gap in between. It has to be the primary context and not one of our
    // own: cudaSetDevice binds the primary one, so anything built in a
    // different context stops working the moment a caller sets the device,
    // including the events a cuBLAS handle keeps inside itself.
    // Losing it is not fatal -- everything but a driver call made before any
    // array operation still works -- so a device that refuses a context is
    // left for that call to report. cuDeviceGet leaves cuDevice untouched
    // when it fails, hence the guard.
    if (cuDeviceGet(&cuDevice, 0) == CUDA_SUCCESS) {
        if (cuDevicePrimaryCtxRetain(&context, cuDevice) == CUDA_SUCCESS) {
            cuCtxSetCurrent(context);
        }
    }
}
