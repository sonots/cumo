# Source code organizations

* `*_kernel.{h,cuh,cu}` files are for device (CUDA kernels).
    * .cu files are compiled via nvcc.
    * .cu files define C wrapper functions to launch CUDA kernels to enable to be called from .c files.
    * Technically, it is not possible to use CRuby API such as `VALUE` in .cu files.
        * CRuby API is not callable from CUDA kernel because they do not have `__device__` modifier.
        * nvcc does not support `#include RUBY_EXTCONF_H`, so can not include `ruby.h`.
    * (RULE) It is allowed to use C++14 codes in .cu files.
* Rest of `*.{h,c}` files are for host (CPU).
    * Call C wrapper functions defined in .cu files.
    * It can use CRuby API.
    * (RULE) It is not allowed to use C++ codes in host files.

Ruby's `mkmf` (or `extconf.rb`) does not support to specify 3rd compiler such as NVCC for another files of extensions `.cu`.
Therefore, cumo specify a wrapper command `bin/mkmf-cu-nvcc` as a compiler and changes its behavor depending on extensions of files to compile.
