# Cumo examples

Ports of the examples in the CuPy repository ([cupy/examples](https://github.com/cupy/cupy/tree/main/examples)) to [Cumo](https://github.com/sonots/cumo).

Every file keeps the directory and name of its CuPy original, with `.py` replaced by `.rb`. Command line options, printed lines and the CPU/GPU structure are kept where Cumo can express them. Where the CPU side of an example uses NumPy, the port uses Numo, and the same method runs on either library, as `cupy.get_array_module` lets it in the original.

## Requirements

* Ruby 3.0 or later
* `numo-narray` or `numo-narray-alt`, for the CPU side of the comparisons
* `cumo`, for the GPU side (only loaded with `GPU=1`). Everything that launches a kernel, uses a stream or passes a struct needs a Cumo newer than 0.9.0
* `numo-linalg` or `numo-linalg-alt`, optional, for BLAS `dot` in `cg` and `gmm` (without it their CPU side is about ten times slower)

## Running

`backend.rb` picks the GPU side the way [narray-llm](https://github.com/Watson1978/narray-llm) does: the `GPU` environment variable set to `1`, `on` or `true` loads Cumo and makes `XM` point at it, and anything else leaves `XM` at Numo and does not load Cumo at all. The CPU side always runs on Numo (`HM`).

```
ruby cg/cg.rb           # CPU side only, no CUDA needed
GPU=1 ruby cg/cg.rb     # CPU side on Numo, then GPU side on Cumo, as the CuPy original
```

Without `GPU=1` the ported examples run their CPU arm and print `Skipping GPU (set GPU=1 to run it on Cumo)` where the GPU arm would be, so they run on a machine without CUDA. The stubs behave the same either way. The examples with no CPU side, which is everything under `jit`, `custom_struct`, `stream` and `peer` plus `finance/monte_carlo*` and `gemm/sgemm`, load Cumo directly and ignore `GPU`. Run an example from its own directory or from the repository root.

## Status

22 of the 32 CuPy examples are ported and run. The 10 that are not need a CUDA library Cumo has no binding for, or a Python library with no Ruby counterpart.

| Example | Status | Notes |
| --- | --- | --- |
| `cg/cg.rb` | Ported | Conjugate gradient, CPU on Numo and GPU on Cumo |
| `kmeans/kmeans.rb` | Ported | Including `--use-custom-kernel`; `--output-image` stops with a notice |
| `gmm/gmm.rb` | Ported | `--output-image` stops with a notice |
| `finance/black_scholes.rb` | Ported | Naive and `ElementwiseKernel` arms |
| `finance/monte_carlo.rb` | Ported | `ElementwiseKernel` with its own xorshift128+ in the preamble |
| `finance/monte_carlo_multigpu.rb` | Ported | `--gpus` takes a comma separated list |
| `gemm/sgemm.rb` | Ported | `sgemm.cu` launched with `Function#launch`, timed with events |
| `jit/*.rb` | Ported | The kernels rewritten in CUDA C, since `cupyx.jit` has no counterpart |
| `custom_struct/*.rb` | Ported | Structs passed by value as `pack`ed bytes, templates by name expression |
| `peer/peer_matrix.rb` | Ported | Prints nothing on a single GPU box, as the original does |
| `stream/cublas.rb` | Ported | |
| `stream/cupy_event.rb` | Ported | |
| `stream/cupy_kernel.rb` | Ported | |
| `stream/cupy_memcpy.rb` | Ported | Pinned host memory |
| `stream/map_reduce.rb` | Ported | Ten streams mapped, one reduce stream waiting on their events |
| `stream/thrust.rb` | Ported | |
| `stream/cufft.rb` | Not ported | No FFT in Cumo |
| `stream/curand.rb` | Ported | `RandomState` and `lognormal` written in the example, over `rand_norm` |
| `stream/cusolver.rb` | Not ported | No cuSOLVER binding in Cumo |
| `stream/cusparse.rb` | Not ported | No sparse matrix or cuSPARSE binding in Cumo |
| `cufft/callback/*.rb` | Not ported | No FFT in Cumo |
| `cutensor/*.rb` | Not ported | No cuTENSOR binding in Cumo |
| `cusparselt/matmul.rb` | Not ported | No cuSPARSELt binding in Cumo |
| `interoperability/mpi4py_multiple_devices.rb` | Not ported | No CUDA-aware MPI for Ruby |

A file marked "Not ported" is a stub. Running it prints which CuPy API the original relies on and what Cumo lacks, then exits with status 1, so the tree stays one-to-one with CuPy's and each stub can be filled in once Cumo grows the feature.

## Differences from the CuPy originals

* The GPU arm is opt-in through `GPU=1`; the CuPy originals always run it.
* Timing uses `cudaDeviceSynchronize` around a monotonic clock, since Cumo has no `Stream.null.synchronize` or CUDA events.
* Reductions in Cumo return 0-dimensional arrays rather than Ruby numbers, so a scalar that steers control flow is read back with `Float(...)` or `Integer(...)`, which reads a Numo scalar too.
* `numpy.random.choice(n, k, replace=False)` is `(0...n).to_a.sample(k)`, and `numpy.where(cond, a, b)` is written as arithmetic on a 0/1 mask, since neither library has a direct counterpart.
* Assertions from `cupy.testing` are written inline.
