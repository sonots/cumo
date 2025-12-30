# 0.5.1 (2025/12/30)

Enhancements:

* Add CUDA 13 support (#153)
* Add cuDNN 9 support

Fixes:

* Backport: fix example code
* Backport: fix example code
* Backport: fix doc
* Backport: fix documents
* Backport: fix document of logseq
* Backport: trim comment out

# 0.5.0 (2025/11/01)

Fixes:

* Remove unnecessary numo-narray dependency
* Fix Errno::EXDEV for Invalid cross-device link
* Remove clobber from default task
* Enable parallel build by default
* Add magic comment for frozen_string_literal
* Backport: fix na_flatten_dim(): SEGV when flattening an empty narray view
* Backport: bug in reshape!: stridx in NArrayView should be reconstructed
* Backport: mask and masked arrays must have the same shape
* Backport: fix na_parse_range() to suppress warnings
* Backport: FIXNUM length is based on LONG, not VALUE
* Backport: fix bug in NArray#sort: qsort() does not support strided loop
* Backport: fix na_aref_md_protected(): na2->stridx should be zero-inizialized
* Backport: q[i].idx should be freed when i != ndim-1
* Backport: fix variable type
* Backport: add tests for Bit view arrays
* Backport: fix macro: STORE_BIT STORE_BIT_STEP: requires mask to leave the lowest bit
* Backport: fix NArray::Bit#any?,all?: empty array should return false
* Backport: fix NArray::Bit#count_true/false: empty array should return zero
* Backport: bug in NArray::Bit; fix bit operation in tmpl_bit/{store_bit,unary,binary}.c
* Fix typo
* Backport 135: Make all empty arrays equal
* Backport: minor fixes in na_get_result_dimension(), check_index_count()
* Backport 116: new method: NArray#fortran_contiguous?
* Backport 186: Fix NMath.sinc(0)
* Backport 188: Fix a typo
* Fix FrozenError
* Use add_dependency instead of add_runtime_dependency
* Remove unused variable
* Remove unused .travis.yml
* Remove unnecessary require to fix warnings of "loading in progress, circular require considered harmful"
* Remove unused variable
* Fix numo-narray library path
* Add extconf_compile_commands_json as development dependency
* Add extconf_compile_commands_json for clangd LSP
* Remove unnecessary loop if disable assert()
* Fix cross-platform negative value conversion for unsigned integer types
* Revert "Fix cross-platform negative value conversion for unsigned integer types"
* Remove unnecessary require
* Add Ractor support
* Update minimum CUDA version
* Update minimum ruby supported version
* Use rake-compiler
* Use absolute file path
* Allow convert nil to NaN in Numo::DFloat.cast
* Fix cross-platform negative value conversion for unsigned integer types
* Fix old-style function definitions
* Fix old-style function definition in qsort.c
* Add required_ruby_version in gemspec
* Use released version of power_assert gem
* Fix LoadError
* Quoted file path
* Add CUDA compute capability (#151)
* extconf.rb: Use File::PATH_SEPARATOR
* Fix build error with cuDNN features
* Link c++ library
* Fix link error with "multiple definition of `cumo_cuda_eCudnnError'"
* Fix failure with Ruby 3.3
* Fix keyword argument expansion
* Remove compute_35 because it was removed at CUDA 12
* Use NVCC_CCBIN env var to detect compiler for cuda code on GCC 15 environment
* Fix build error with GCC 15
* Use rb_cObject instead of rb_cData
* Remove unnecessary dependency
* at() method was rewritten in C.

# 0.4.3 (2019-06-11)

Fixes:

* Fix max|min\_index to behave like numo with CUMO\_COMPATIBLE\_MODE=ON

# 0.4.2 (2019-06-11)

Fixes:

* cond_unary.c: add cudaDeviceSynchronize to avoid bus error
* index.c: add cudaDeviceSynchronize to avoid bus error
* cum.c: add cudaDeviceSynchronize to avoid bus error

# 0.4.1 (2019-05-06)

Fixes:

* Fix `fixed_batch_norm`

# 0.4.0 (2019-05-04)

Released (same with 0.3.5)

# 0.3.5 (2019-05-04)

Fixes:

* Fix `each_with_index` to synchronize on each element

# 0.3.4 (2019-05-04)

Enhancements:

* Support cuDNN fixed\_batch\_norm (cudnnBatchNormalizationForwardInference)

# 0.3.3 (2019-05-02)

Fixes:

* Fix each to synchronize on each element

# 0.3.2 (2019-05-02)

Fixes:

* Fix max and max\_index for sfloat and dfloat

# 0.3.1 (2019-04-16)

Fixes:

* Fix batch\_norm\_backward
* Fix scalar.dot(scalar)
* Fix clip

# 0.3.0 (2019-04-10)

Enhancements:

* Support cuDNN
  * conv (cudnnConvolution)
  * conv\_transpose (cudnnConvolutionBackwardData)
  * conv\_grad\_w (cudnnConvolutionBackwardFilter)
  * batch\_norm (cudnnBatchNormalizationForwardTraining)
  * batch\_norm\_backward (cudnnBatchNormalizationBackward)
  * avg\_pool and max\_pool (cudnnPoolingForward)
  * avg\_pool\_backward and max\_pool\_backward (cudnnPoolingBackward)

# 0.2.5 (2019-03-04)

Enhancements:

* Support arithmetic sequence, which is available in ruby >= 2.6.0 (thanks to naitoh)

# 0.2.4 (2018-11-21)

Changes:

* Turn off `CUMO_SHOW_WARNING` as default

# 0.2.3 (2018-11-17)

Enhancements:

* Add some missing `synchronize` workarounds

# 0.2.2 (2018-11-13)

Enhancements:

* CUDA kernelize na\_index\_aref\_naview
* CUDA kernelize na\_index\_aref\_nadata
* CUDA kernelize diagonal
* CUDA kernelize copy

# 0.2.1 (2018-11-12)

Enhancements:

* Add `CUMO_SHOW_WARNING` and `CUMO_SHOW_WARNING_ONCE` environment variables to suppress cumo warnings (They are only for debug purpose, would be removed in future).

# 0.2.0 (2018-11-12)

All tests in red-chainer passed.

Fixes:

* Fix advanced indexing
* Fix accum\_index reduction (max\_index, and min\_index)
