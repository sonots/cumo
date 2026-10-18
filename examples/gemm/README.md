# SGEMM example

This example contains implementation of single-precision general matrix-multiplication  (SGEMM).
The implementation is based on the one in [MAGMA](http://icl.cs.utk.edu/magma/).

`sgemm.rb` compiles the kernel in `sgemm.cu`, kept unchanged, and launches it with `Cumo::CUDA::Function#launch`. `utils.rb` times it and cuBLAS with CUDA events.


### How to demo
The demo contains a script that calculates matrix multiplication of A (m x k) and B (k x n).
The demo can be run by the following command.

```
ruby sgemm.rb [--gpu GPU_ID] [--m m] [--n n] [--k k]
```


### What this demo contains

In this example, we work on a SGEMM kernel that requires a complete interface to `cuLaunchKernel` (e.g. grid size and size of shared memory), which is not provided by `cupy.ElementwiseKernel`.
CuPy arrays work regardless of the underlying memory layouts thanks to `ndarray` abstraction.
As is the case for this example, `ndarray` abstraction does not need to be used if the underlying memory layouts of arrays match the ones expected by a kernel.
The SGEMM kernel expects input and output arrays to be in Fortran contiguous memory layout, and this layout is enforced by `cupy.asfortranarray`. Cumo has no Fortran order, so `as_fortran` in `sgemm.rb` makes a transposed copy instead.

#### How to dynamically compile and launch a kernel function written in CUDA C

For compilation, `cupy.RawKernel` class is used to compile a CUDA code written in `sgemm.cu`.
The class takes a text of code and name of the kernel as an constructor argument.
The instance is a callable; the CUDA code will be compiled and then invoked when it is called.
The compiled code is cached, and it avoids the compilation process after the first time.
Also, the CUDA code can be modified at Python level because it is simply a text.
In this example, C macros that determine a distribution of data to threads are specified at runtime.
Note that `"extern C"` needs to be put on top of the kernel that is called.

Cumo has the compile half of this: `Cumo::CUDA::Compiler#compile_with_cache` runs NVRTC, links the PTX and loads the cubin into a `Cumo::CUDA::Module`, with the same on-disk cache. `read_code` in `utils.rb` prepends the macros the same way.

#### How to supply grid size, block size and shared memory size on launching a kernel function

`cupy.RawKernel` object allows you to call the kernel with CUDA's `cuLaunchKernel` interface.
In other words, you have control over grid size, block size, shared memory size and stream.
At this level of interface, it becomes straightforward to replace host `.cu` that calls CUDA kernels with Python code.

Cumo has this half too: `Function#launch(args, grid:, block:, shared_mem:, stream:)` takes the same four, and `sgemm.rb` passes the grid, the block and the shared memory size.
