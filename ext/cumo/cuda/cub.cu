#include <cub/util_device.cuh>

namespace {

__global__ void cumo_cub_ptx_version_probe() {}

}  // namespace

// CUB learns a device's PTX version once, from the attributes of an empty
// kernel. Every translation unit has its own empty kernel, but the link keeps
// one copy of the code that asks, and it was the one in the sort module, so the
// first scan of a process loaded every sort kernel to answer it. Answering from
// this module's kernel, built with the same flags, gives the same version.
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
