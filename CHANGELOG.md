# 0.5.10 (2026/08/23)

Breaking changes:

* An indexed assignment to a frozen view raises instead of writing through it; `store` and `fill` already raised (PR #306)
* The cuDNN and cuBLAS entry points turn away arguments they used to read or write out of bounds: `pooling_backward` requires `y` and `gy` to carry its own dtype and the pooling output shape of `x`, `gemm` requires a contiguous `c`, `batch_norm` requires the reduced size to cover x's channels, and `conv_grad_w` raises on a mismatched `gy` rather than asserting (PR #303, PR #302, PR #300, PR #278)
* `seq` on the unsigned types wraps a negative start and a start past 2**63 the way `fill` and `cast` do, where they collapsed to zero and clamped to INT64_MAX (PR #295, PR #294)
* A negative power of an unsigned array answers zero instead of spinning the device until SIGKILL (PR #293)
* A subscript that is a Cumo::RObject subclass, and an `expand_dims` past `CUMO_NA_MAX_DIMENSION`, raise instead of being taken (PR #272, PR #270)

Fixes:

* Fix `[]=` writing through a frozen view, the store going to a derived view whose data object is the root rather than the receiver (PR #306)
* Fix `at()` reading the accumulator in place of the subscript on a reversed view, which addressed far in front of the buffer (PR #305)
* Fix `cumo_cuda_cudnn_CreateBNTensorDescriptor` returning before it derived the descriptor, an inverted status check that no caller reached (PR #304)
* Fix `batch_norm`, `fixed_batch_norm` and `batch_norm_backward` reading and writing past their parameters, the sizes being checked against the reduced shape while cuDNN reaches x's channel count (PR #303)
* Fix `gemm` writing past its allocation when `c` is a non-contiguous inplace view, and over whatever the pool put next to it (PR #302)
* Fix `pooling_backward` reading past `y` and `gy`, and returning what it read in `gx` (PR #300)
* Fix `pow_int` negating INT32_MIN, which overflows and leaves the loop unbounded (PR #299)
* Fix the device free running for a subscript that allocated no index array (PR #297)
* Fix a NULL dereference from `at()` given a scalar subscript (PR #271)
* Fix an out-of-bounds write through an unchecked cuDNN output array (PR #269)
* Fix `gemm` reading past an operand that carries fewer batch dimensions (PR #268)

Changes:

* Pass a numeric operand to the kernel instead of casting it to a 0-dimensional array first, in the operators, `clip`, `pow`, the comparisons, the NMath functions and the coerced left-hand side; GPT-2 124M decode drops from 787 to 677 kernel launches a token (PR #296, PR #291, PR #290, PR #289, PR #288, PR #287)
* Run the NMath binary functions through the indexer loop instead of one kernel launch per row; 11x on a transposed operand and up to 60x on the cheap kernels (PR #292)
* Copy a reduction operand only when it carries an index array, where any non-contiguous view was copied whole (PR #286)
* Run the nan-aware reductions, the nan-aware index reductions, `kahan_sum` and `Bit#mask` on the GPU (PR #277, PR #276, PR #275, PR #273)
* Take a transposed batch of matrices with the cuBLAS transpose flag rather than duplicating it, which saves a temporary the size of the operand (PR #283)
* Emit indexer accessors up to eight dimensions (PR #284)
* Drop a free-list bin from the pool arena when `Malloc` empties it (PR #279)
* Check the cuDNN header and version, not just the library (PR #280)
* Document every method that returns a 0-dimensional NArray, and what keeping scalars on the device buys (PR #298, PR #282)
* Build two GPU architectures in CI instead of every one (PR #285)
* Add `fiddle` to the Gemfile (PR #281)
* Remove TODO comments whose questions have been answered (PR #274)
* Cover the Int32 array exponent in the INT_MIN power test (PR #301)

# 0.5.9 (2026/08/20)

Breaking changes:

* `sort_index` resolves ties to the lowest index, the GPU sort being stable, where the host quicksort left them in whatever order it produced (PR #261)

Fixes:

* Fix `a[[2, 1], true].store(array)` writing every row onto row 0, on every dtype: ndloop read the index array on the host before the kernel that fills it had run (PR #265)
* Fix a segmentation fault storing a Ruby Array into a reversed `Cumo::Bit` view that starts past the first word (PR #266)
* Fix `sort` in place of a view backed by an index array reading row 0 for every row (PR #262)
* Fix a `Cumo::RObject` store reading an index array before the kernel writes it, which answered zeroes from the second store onwards (PR #258)
* Fix a store of an Array of narrays into an indexed destination writing one element per row (PR #257)
* Fix the zero fill of a row stored from a shorter narray starting one element too far (PR #256)
* Fix grid-stride kernels hanging from about 2**32 elements, the step wrapping to zero (PR #240)
* Fix `Cumo::Bit` views with a negative step reading out of bounds (PR #238)
* Fix index reductions answering a later index on ties, where `Numo::NArray` answers the first (PR #236)
* Fix wrong values from reductions with `nan: true` (PR #234)

Changes:

* Run `sort`, `sort_index` and `median` on the GPU with `cub::DeviceSegmentedRadixSort`; 1M `SFloat` `sort` 115.7 -> 0.47 ms, `median` 107.8 -> 0.13 ms, `sort_index` 115.8 -> 1.69 ms (PR #259, PR #260, PR #261)
* Run `poly` on the GPU, which had called its iterator once per element behind a synchronization; 2^20 `DFloat` 163.8 -> 0.021 ms (PR #263)
* Run `bincount` on the GPU; 1M `Int32` 1.62 -> 0.05 ms (PR #255)
* Run `cumsum` and `cumprod` on the GPU (PR #242)
* Run `minmax`, `abs`, `isnan` and the rest of `cond_unary`, `modf` and `frexp` on the GPU (PR #230, PR #231, PR #232, PR #233)
* Run the `Cumo::Bit` operators, reductions, `where`, `where2`, `fill` and the stores on the GPU (PR #237, PR #239, PR #241, PR #252, PR #254, PR #264, PR #266)
* Run complex `real=` and `imag=` on the GPU; 2^22 `DComplex` 4.63 -> 0.45 ms (PR #267)
* Run the accumulating reductions and `mulsum` on more than one GPU thread, where they had been running device-side thrust in a single thread (PR #243, PR #244)
* Address a whole non-contiguous view in one kernel launch instead of one per row, in the elementwise templates, the stores, the copies and the Bit operands (PR #245, PR #246, PR #249, PR #251, PR #252)
* Read a flat reduction operand without the indexer, and reduce over an outer axis with host-side addressing; `sum(axis: 0)` on 4096x4096 2.01 -> 0.017 ms (PR #248, PR #253)
* Split a reduction with too few outputs across more blocks (PR #235)
* Build the C++ and CUDA sources with optimization: mkmf leaves `$(optflags)` out of `CXXFLAGS`, so every `.cpp` and the host half of every `.cu` had been built at `-O0`; `a + b` 2.86 -> 2.00 us (PR #250)
* Warn about the synchronization `a[idx]` and `inspect` perform (PR #247)
* Add benchmark scripts under `bench/` for a transformer block, k-means, a 2D Ising model, conjugate gradient, a particle simulation and the CPU/GPU crossover, and two probes for calls that synchronize or launch one kernel per row

# 0.5.8 (2026/08/17)

Breaking changes:

* Random numbers are generated on the GPU, so `srand` no longer reproduces `Numo::NArray`'s values for the same seed. Reproducibility within Cumo is unchanged and no longer depends on how the calls are split (PR #223)
* Integer division by zero raises `ZeroDivisionError` instead of returning whatever the hardware produced, matching `Numo::NArray` (PR #218)
* A numeric index into an unallocated NArray raises instead of returning a view nothing ever reads (PR #225)

Fixes:

* Fix `rb_raise` from inside a C++ catch leaking the exception object (PR #229)
* Fix a failed Array store leaking its staging buffer, as much as the destination holds per failure (PR #228)
* Fix the GC free hook raising when a device free fails, which surfaced at whatever line triggered the collection (PR #227)
* Fix `from_binary` and `store_binary` writing managed memory while a kernel was still reading it (PR #226)
* Fix integer `sum` and `prod` truncating every partial the reduction merged (PR #224)
* Fix `minmax` ignoring compatible mode and returning zero-dimensional NArrays (PR #220)
* Fix `nan` poisoning `max` and an all-`nan` `min` answering `DBL_MAX` (PR #219)
* Fix `reshape!` and `marshal_load` mutating the array before they validate their arguments (PR #217)
* Fix `ptp` answering 1 whatever the input (PR #216)
* Fix `mulsum` ignoring the accumulator and the operand strides, which faulted the GPU on 8-bit dtypes (PR #215)
* Fix `dot` handing `gemm` operands of a different dtype, an out-of-bounds read (PR #214)
* Fix view offsets being added in the wrong unit (PR #213)
* Fix interpreter abort on a zero-dimensional view (PR #212)
* Fix `Cumo::RObject#logseq` returning `Infinity` (PR #210)
* Fix `to_i`, `to_f` and `to_c` recursing forever on a one-element array (PR #208)
* Fix complex `log2` and `log10` discarding the logarithm on the GPU (PR #207)
* Fix segfault in `max_index` and `min_index` with `nan: true` (PR #206)
* Fix `poly` failing on every input from an uninitialized `ndfunc` dimension (PR #205)
* Fix out-of-bounds writes in `bincount` from an overflowing length and a stale scan (PR #204)
* Fix `bincount` raising `TypeError` for every input (PR #203)
* Fix host loops reading device memory without synchronizing, which silently corrupted `minmax`, the `nan: true` reductions, `kahan_sum`, `modf`, `frexp` and `set_imag` (PR #202)
* Fix out-of-bounds access from three unchecked `size_t` multiplications (PR #201)
* Fix segfaults from an unvalidated marshal payload (PR #200)
* Fix `store_array` reading a source Array that its own elements rewrite (PR #198)
* Fix CUDA initialisation statuses being ignored at `require` time (PR #196)
* Fix kernel launch errors being discarded, which let a rejected launch return an untouched buffer as success (PR #195)
* Fix wrong `Cumo::RObject` results from driving host memory with asynchronous CUDA work (PR #194)
* Fix device memory being invisible to the GC, which let a program churning temporaries run the GPU out of memory (PR #221)

Changes:

* Run `rand` and `rand_norm` on the GPU (PR #223)
* Run `clip` on the GPU (PR #222)
* Correct spelling in messages, docs and comments (PR #211)
* Add the math, extra and narray test suites `numo-narray-alt` has and cumo lacked (PR #207, PR #208, PR #209)
* Add a `store_array` regression test for a shrunk source Array (PR #199)
* Stop the ccache cache growing past the 10 GB limit in CI (PR #197)
* Add a benchmark script under `bench/` (commit 3127f84)

# 0.5.7 (2026/08/13)

Breaking changes:

* `reshape` raises `ArgumentError` instead of `RangeError` for a dimension above `INT_MAX` (PR #183)
* `parse` accepts only numeric literals and `true`/`false`/`nil`, so `parse("3/4")` raises `ArgumentError` instead of returning `[[0]]` (PR #190)

Fixes:

* Fix segfaults from CUDA handles the process did not create, and from destroying one twice (PR #192)
* Fix process abort from freeing NArray data with the wrong allocator, including a raise out of GC for `Cumo::RObject` (PR #191)
* Fix arbitrary code execution in `parse`, which ran `eval` on every token (PR #190)
* Fix stack buffer overflow in `Cumo::RObject#format` with an element longer than 47 characters (PR #189)
* Fix a freed chunk returning to the arena of the wrong stream (PR #187)
* Fix `Malloc` searching the free list with the default stream's index (PR #186)
* Fix data race on the memory pool chunk graph and on the per-device pool table (PR #185)
* Fix a SIGFPE on fractional range steps and an out-of-bounds read on newaxis (PR #184)
* Fix a SIGFPE and a silently corrupt shape from unchecked `reshape` arguments (PR #183)
* Fix segfault when `store_binary` receives a non-String (PR #182)
* Fix missing range check on NArray indices, which segfaulted on a negative one (PR #181)

Changes:

* Compile the memory pool test harness in CI, and run it from `rake test` (PR #188)

# 0.5.6 (2026/08/09)

Fixes:

* Fix out-of-tree build failing to create the narray/types object directory
* Fix segfaults from unchecked String arguments in Driver (#179)
* Fix out-of-bounds read of include_names in nvrtcCreateProgram (#178)
* Fix store into a strided view reading a released staging buffer (#177)
* Fix pinned host memory leak when indexing by an array (#176)
* Fix segfault in Module#get_global_var reading device memory as a host pointer (#174)
* Fix freeing pooled memory with cudaFree after disabling the pool (#173)
* Fix run-ctest failing to start the test binary (#172)
* Fix process abort when freeing device memory fails (#171)
* Fix memory pool handing out a tiny chunk for a huge allocation (#170)
* Fix out-of-bounds read of a shorter sub-narray in store (#169)
* Fix out-of-bounds device write in store when a sub-narray is too long (#168)
* Backport: fix na_flatten_dim for multi-dimensional empty arrays (#167)
* Backport: make qsort loop condition explicit to prevent incorrect optimization (#166)
* Backport: free previously allocated shape in cumo_na_alloc_shape (#165)
* Backport: free shape on deallocation regardless of size (#164)
* Backport: convert max to double in rand method of Cumo::RObject (#163)
* Backport: use an inline function to prevent double use of macro argument (#162)
* Fix out-of-bounds stridx access in at() when :new is given (#161)
* Backport: prevent out-of-bounds access to stridx when orig_dim exceeds ndim (#160)
* Backport: prevent negative array index in na_get_strides_nadata when ndim is zero (#159)
* Backport: use correct array index for mod result in divmod macro for Cumo::RObject (#158)

# 0.5.5 (2026/07/11)

Fixes:

* Backport: rename method: min_arg, max_arg => argmin, argmax
* Backport: fix NArray#concatenate: avoid exception when concatenating empty arrays
* Backport: fix boolean indexing in each axis

# 0.5.4 (2026/05/30)

Fixes:

* Backport: Fix to use rb_funcallv_kw in Ruby 2.7 and later
* Backport: define id_pow as "**" for all types except Bit
* Backport: define id_minus as "-@" for RObject
* Backport: duplicated id_eq and id_ne for RObject
* Backport: follow the change of INIT_PTR_BIT_IDX
* Backport: add test for empty bit array
* Backport: fix NArray::Bit#count_true/false: empty array should return zero
* Fix backport error
* Backport: minor fixes
* Backport: fix RObject#sum,prod
* Backport: NArray#from_binary: allow zero dimension
* Backport: Added test cases
* Backport: fix NArray#store: drop automatic 'to_a' conversion
* Backport: Numo::NArray.cast tries to convert to Array when it has to_a method
* Backport: use rb_respond_to()
* Backport: improve discrimination between int32/64
* Backport: missing declaration of id_ge
* Backport: support casting any object that responds to 'to_a'
* Backport: fix the range of int32 as -2**31 .. 2**31-1
* Backport: if int32_max is Bignum, it should be protected from GC
* fix: handle compiler flags included in RbConfig::CONFIG properly (#157)
* Backport: Added more percentile tests
* Backport: Use File.write in cogen to ensure file is closed
* Backport: Added percentile method
* Backport: Use INT2FIX to define m_data_to_num for 8 or 16-bit Int.
* Backport: raise error if compiler does not support  8 or 4-byte integer

# 0.5.3 (2026/04/26)

Fixes:

* Fix unsupported gpu architecture errors in CUDA 12.8 and 12.9 (#156)

# 0.5.2 (2026/01/25)

Fixes:

* Backport: Add support for copy on write with store_binary and frozen string
* Remove unnecessary debug code
* Fix capability list
* Build only with supported capabilities to reduce compilation time
* Fix SEGV when calling {mean, var, stddev, rms} on a single-element array (#154)
* Suppress warning message for deprecated declarations
* Fix variable typo in complex log2 and log10 functions (#152)

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
