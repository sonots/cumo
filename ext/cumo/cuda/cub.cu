#include <cub/util_device.cuh>
#include "cumo/template_kernel.h"

namespace {

__global__ void cumo_cub_ptx_version_probe() {}

}  // namespace

// CUB learns each device's PTX version once, from the attributes of an empty
// kernel. Each translation unit has its own empty kernel, but the link keeps
// one copy of the code that asks. The query loads the module that copy came
// from, which can be a large one the caller never runs. This module has one
// kernel built with the same flags, so it gives the same version at little cost.
extern "C" void
cumo_cuda_cub_prime_ptx_version(void)
{
    cub::GetPerDeviceAttributeCache<cub::PtxVersionCacheTag>()(
        [](int& ptx_version) {
            cudaFuncAttributes attrs = {};
            cudaError_t status = cudaFuncGetAttributes(&attrs, (const void*)cumo_cub_ptx_version_probe);
            ptx_version = attrs.ptxVersion * 10;
            return status;
        },
        cub::CurrentDevice());
}
