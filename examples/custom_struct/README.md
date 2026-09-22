# Custom user structure examples

This folder contains examples of custom user structures in `cupy.RawKernel` (see [https://docs.cupy.dev/en/stable/tutorial/kernel.html](https://docs.cupy.dev/en/stable/tutorial/kernel.html) for corresponding documentation).

**Status: not ported.** All three scripts stop with a message. They need `cupy.RawKernel` / `cupy.RawModule` (Cumo cannot launch a kernel) and NumPy structured dtypes to lay out the host side of a struct (Cumo has no equivalent). See `../MISSING_FEATURES.md`.

This folder provides three scripts ranked by increasing complexity:

1. `builtin_vectors.rb` shows how to use CUDA builtin vectors such as `float4` both as scalar parameter (pass by value from host) and array parameter in RawKernels.
2. `packed_matrix.rb` demonstrates how to create and use templated packed structures in RawModules.
3. `complex_struct.rb` illustrates the possibility to recursively build complex NumPy dtypes matching device structure memory layout.

All examples can be run as simple ruby scripts: `ruby example_name.rb`.
