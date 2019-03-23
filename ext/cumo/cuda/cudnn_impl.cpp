#include "cumo/cuda/cudnn.h"

#include <assert.h>
#include <ruby.h>
#include <cudnn.h>
#include "cumo/narray.h"
#include "cumo/template.h"
#include "cumo/cuda/runtime.h"
#include "cumo/cuda/memory_pool.h"

#include <unordered_map>

#if defined(__cplusplus)
extern "C" {
#if 0
} /* satisfy cc-mode */
#endif
#endif

 size_t
cumo_cuda_cudnn_GetConvOutDim(size_t in_dim, size_t kernel_size, size_t stride, size_t pad) {
    // assert(stride > 0);
    int64_t numerator;
    // if (cover_all) {
    //     numerator = in_dim + pad * 2 - kernel_size + stride - 1;
    // } else {
    numerator = in_dim + pad * 2 - kernel_size;
    // }
    // if (numerator < 0) {
    //     throw DimensionError{"Output size should be positive."};
    // }
    return (size_t)(numerator / stride + 1);
}

cudnnStatus_t
cumo_cuda_cudnn_CreateTensorDescriptor(
        cudnnTensorDescriptor_t *desc,
        VALUE a, cudnnDataType_t cudnn_dtype) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cumo_narray_t *na;
    CumoGetNArray(a, na);
    int ndim = (int)(na->ndim);
    size_t *shape = na->shape;

    assert(cumo_na_check_contiguous(a) == Qtrue);
    status = cudnnCreateTensorDescriptor(desc);
    if (status != CUDNN_STATUS_SUCCESS) return status;

    if (ndim == 4) {
        status = cudnnSetTensor4dDescriptor(
                *desc, CUDNN_TENSOR_NCHW, cudnn_dtype, shape[0], shape[1], shape[2], shape[3]);
    }
    else {
        int int_shape[CUMO_NA_MAX_DIMENSION];
        for (int idim = 0; idim < ndim; ++idim) {
            int_shape[idim] = (int)(shape[idim]);
        }
        int int_strides[CUMO_NA_MAX_DIMENSION]; // strides divided by item size
        int stride = 1;
        for (int idim = ndim - 1; idim >= 0; --idim) {
            int_strides[idim] = stride;
            stride *= int_shape[idim];
        }
        status = cudnnSetTensorNdDescriptor(*desc, cudnn_dtype, ndim, int_shape, int_strides);
    }
    return status;
}

cudnnStatus_t
cumo_cuda_cudnn_CreateFilterDescriptor(
        cudnnFilterDescriptor_t *desc,
        VALUE a,
        cudnnDataType_t cudnn_dtype) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cumo_narray_t *na;
    int ndim;
    size_t *shape;

    CumoGetNArray(a, na);
    ndim = (int)(na->ndim);
    shape = na->shape;

    assert(cumo_na_check_contiguous(a) == Qtrue);
    status = cudnnCreateFilterDescriptor(desc);
    if (status != CUDNN_STATUS_SUCCESS) return status;

    if (ndim == 4) {
        status = cudnnSetFilter4dDescriptor(
                *desc, cudnn_dtype, CUDNN_TENSOR_NCHW, shape[0], shape[1], shape[2], shape[3]);
    } else {
        int int_shape[CUMO_NA_MAX_DIMENSION];
        for (int idim = 0; idim < ndim; ++idim) {
            int_shape[idim] = (int)(shape[idim]);
        }
        status = cudnnSetFilterNdDescriptor(*desc, cudnn_dtype, CUDNN_TENSOR_NCHW, ndim, int_shape);
    }

    return status;
}

cudnnStatus_t
cumo_cuda_cudnn_CreateConvolutionDescriptor(
        cudnnConvolutionDescriptor_t *desc,
        size_t ndim, int* int_stride, int* int_pad,
        cudnnDataType_t cudnn_dtype) {
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    int int_dilation[CUMO_NA_MAX_DIMENSION];
    for (size_t idim = 0; idim < ndim; ++idim) {
        int_dilation[idim] = 1;
    }

    status = cudnnCreateConvolutionDescriptor(desc);
    if (status != CUDNN_STATUS_SUCCESS) return status;

    if (ndim == 2) {
        status = cudnnSetConvolution2dDescriptor(
                *desc,
                int_pad[0],
                int_pad[1],
                int_stride[0],
                int_stride[1],
                int_dilation[0],
                int_dilation[1],
                CUDNN_CROSS_CORRELATION,
                cudnn_dtype);
    } else {
        status = cudnnSetConvolutionNdDescriptor(
                *desc, ndim, int_pad, int_stride, int_dilation, CUDNN_CROSS_CORRELATION, cudnn_dtype);
    }

    return status;
}

// Borrowed from boost::hash_combine
//
// TODO(sonots): hash combine in 64bit
static void HashCombine(std::size_t& seed, std::size_t hash_value) {
    seed ^= hash_value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Partially Borrowed from ChainerX
struct AlgoCacheKey {
    size_t ndim;  // # of spatial dimensions
    size_t x_shape[CUMO_NA_MAX_DIMENSION];
    size_t w_shape[CUMO_NA_MAX_DIMENSION];
    size_t y_shape[CUMO_NA_MAX_DIMENSION];
    size_t pad[CUMO_NA_MAX_DIMENSION];
    size_t stride[CUMO_NA_MAX_DIMENSION];
    cudnnDataType_t dtype;
    size_t max_workspace_size;

    bool operator==(const AlgoCacheKey& other) const {
        if (ndim != other.ndim) return false;
        if (dtype != other.dtype) return false;
        if (max_workspace_size != other.max_workspace_size) return false;
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            if (x_shape[idim] != other.x_shape[idim]) return false;
        }
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            if (w_shape[idim] != other.w_shape[idim]) return false;
        }
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            if (y_shape[idim] != other.y_shape[idim]) return false;
        }
        for (size_t idim = 0; idim < ndim; ++idim) {
            if (pad[idim] != other.pad[idim]) return false;
        }
        for (size_t idim = 0; idim < ndim; ++idim) {
            if (stride[idim] != other.stride[idim]) return false;
        }
        return true;
    }

    bool operator!=(const AlgoCacheKey& other) const { return !operator==(other); }
};

struct AlgoCacheKeyHash {
    using result_type = std::size_t;
    std::size_t operator()(const AlgoCacheKey& key) const {
        std::size_t seed = 0;
        size_t ndim = key.ndim;
        HashCombine(seed, std::hash<size_t>()(key.ndim));
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            HashCombine(seed, std::hash<size_t>()(key.x_shape[idim]));
        }
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            HashCombine(seed, std::hash<size_t>()(key.w_shape[idim]));
        }
        for (size_t idim = 0; idim < ndim + 2; ++idim) {
            HashCombine(seed, std::hash<size_t>()(key.y_shape[idim]));
        }
        for (size_t idim = 0; idim < ndim; ++idim) {
            HashCombine(seed, std::hash<size_t>()(key.pad[idim]));
        }
        for (size_t idim = 0; idim < ndim; ++idim) {
            HashCombine(seed, std::hash<size_t>()(key.stride[idim]));
        }
        HashCombine(seed, std::hash<int>()((int)(key.dtype)));
        HashCombine(seed, std::hash<size_t>()(key.max_workspace_size));
        return seed;
    }
};

using FwdAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionFwdAlgo_t, size_t>, AlgoCacheKeyHash>;
using BwdDataAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionBwdDataAlgo_t, size_t>, AlgoCacheKeyHash>;
using BwdFilterAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t>, AlgoCacheKeyHash>;

// TODO: Another cache for another device
static FwdAlgoCacheMap fwd_algo_cache_map_{};
static BwdDataAlgoCacheMap bwd_data_algo_cache_map_{};
static BwdFilterAlgoCacheMap bwd_filter_algo_cache_map_{};

cudnnStatus_t
cumo_cuda_cudnn_FindConvolutionForwardAlgorithm(
        cudnnConvolutionFwdAlgoPerf_t *perf_result,
        cudnnHandle_t handle,
        cudnnTensorDescriptor_t x_desc,
        VALUE x,
        cudnnFilterDescriptor_t w_desc,
        VALUE w,
        cudnnConvolutionDescriptor_t conv_desc,
        cudnnTensorDescriptor_t y_desc,
        VALUE y,
        size_t max_workspace_size,
        int* int_stride,
        int* int_pad,
        size_t ndim,
        cudnnDataType_t cudnn_dtype)
{
    cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
    cumo_narray_t *nx, *nw, *ny;
    CumoGetNArray(x, nx);
    CumoGetNArray(w, nw);
    CumoGetNArray(y, ny);

    auto key = AlgoCacheKey{};
    key.ndim = ndim;
    for (size_t idim = 0; idim < ndim + 2; ++idim) {
        key.x_shape[idim] = nx->shape[idim];
        key.w_shape[idim] = nw->shape[idim];
        key.y_shape[idim] = ny->shape[idim];
    }
    for (size_t idim = 0; idim < ndim; ++idim) {
        key.pad[idim]= int_pad[idim];
        key.stride[idim]= int_stride[idim];
    }
    key.dtype = cudnn_dtype;
    key.max_workspace_size = max_workspace_size;

    auto& algo_cache_map = fwd_algo_cache_map_;
    // TODO: thread-safe
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        auto pair = it->second;
        perf_result->algo = pair.first;
        perf_result->memory = pair.second;
        return CUDNN_STATUS_SUCCESS;
    }

    char* x_ptr = cumo_na_get_pointer_for_read(x) + cumo_na_get_offset(x);
    char* w_ptr = cumo_na_get_pointer_for_read(w) + cumo_na_get_offset(w);
    char* y_ptr = cumo_na_get_pointer_for_read(y) + cumo_na_get_offset(y);

    char* workspace = cumo_cuda_runtime_malloc(max_workspace_size);
    int returned_algo_count{};
    status = cudnnFindConvolutionForwardAlgorithmEx(
                handle,
                x_desc,
                (void*)x_ptr,
                w_desc,
                (void*)w_ptr,
                conv_desc,
                y_desc,
                (void*)y_ptr,
                1,  // requested algo count,
                &returned_algo_count,
                perf_result,
                (void*)workspace,
                max_workspace_size);
    cumo_cuda_runtime_free(workspace);
    if (status != CUDNN_STATUS_SUCCESS) return status;
    assert(returned_algo_count == 1);

    // TODO: thread-safe
    algo_cache_map[key] = {perf_result->algo, perf_result->memory};
    return status;
}

#if defined(__cplusplus)
#if 0
{ /* satisfy cc-mode */
#endif
}  /* extern "C" { */
#endif
