# Custom user structure examples

This folder contains examples of custom user structures in `cupy.RawKernel` (see [https://docs.cupy.dev/en/stable/tutorial/kernel.html](https://docs.cupy.dev/en/stable/tutorial/kernel.html) for corresponding documentation).

The ports compile each kernel with `Cumo::CUDA::Compiler#compile_with_cache` and launch it with `Cumo::CUDA::Function#launch`. Cumo has no structured dtypes, so a struct passed by value is the `pack`ed bytes of its fields, and a template kernel is picked by its name expression.

This folder provides three scripts ranked by increasing complexity:

1. `builtin_vectors.rb` shows how to use CUDA builtin vectors such as `float4` both as scalar parameter (pass by value from host) and array parameter in a kernel.
2. `packed_matrix.rb` demonstrates how to create and use templated packed structures in a module compiled with name expressions.
3. `complex_struct.rb` builds the bytes of a nested struct from the member offsets and sizes that the device reports.

All examples can be run as simple ruby scripts: `ruby example_name.rb`.
