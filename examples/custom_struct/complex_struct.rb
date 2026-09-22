# frozen_string_literal: true

require 'cumo/narray'

STRUCT_DEFINITION = <<~CUDA
  struct complex_struct {
      int4 a;
      char b;
      double c[2];
      short1 d;
      unsigned long long int e[3];
  };
CUDA

STRUCT_LAYOUT_CODE = <<~CUDA
  #{STRUCT_DEFINITION}

  extern "C" __global__ void get_struct_layout(
                                  unsigned long long *itemsize,
                                  unsigned long long *sizes,
                                  unsigned long long *offsets) {
      const complex_struct* ptr = nullptr;

      itemsize[0] = sizeof(complex_struct);

      sizes[0] = sizeof(ptr->a);
      sizes[1] = sizeof(ptr->b);
      sizes[2] = sizeof(ptr->c);
      sizes[3] = sizeof(ptr->d);
      sizes[4] = sizeof(ptr->e);

      offsets[0] = (unsigned long long)&ptr->a;
      offsets[1] = (unsigned long long)&ptr->b;
      offsets[2] = (unsigned long long)&ptr->c;
      offsets[3] = (unsigned long long)&ptr->d;
      offsets[4] = (unsigned long long)&ptr->e;
  }
CUDA

KERNEL_CODE = <<~CUDA
  #{STRUCT_DEFINITION}

  extern "C" __global__ void test_kernel(const complex_struct s,
                                         double* out) {
      int i = threadIdx.x;
      double sum = 0.0;
      sum += s.a.x + s.a.y + s.a.z + s.a.w;
      sum += s.b;
      sum += s.c[0] + s.c[1];
      sum += s.d.x;
      sum += s.e[0] + s.e[1] + s.e[2];
      out[i] = i * sum;
  }
CUDA

def assert_array_almost_equal(actual, desired, decimal: 6)
  diff = (Cumo::DFloat.cast(actual) - Cumo::DFloat.cast(desired)).abs.max
  return if Float(diff) < 1.5 * (10**-decimal)

  raise "Arrays are not almost equal to #{decimal} decimals: largest difference #{Float(diff)}"
end

def main
  # This program demonstrate how to build a hostside
  # representation of device structure 'complex_struct'
  # defined in STRUCT_DEFINITION that can be used as a kernel argument.

  # First step is to determine structure memory layout
  #  itemsize -> overall struct size
  #  sizes    -> individual struct member sizes, determined with sizeof
  #  offsets  -> individual struct member offsets, determined with offsetof
  # Results (in terms of bytes) are copied to host after kernel launch.
  # Note that 'complex_struct' has 5 members named a, b, c, d and e.
  itemsize = Cumo::UInt64.new(1)
  sizes = Cumo::UInt64.new(5)
  offsets = Cumo::UInt64.new(5)

  kernel = Cumo::CUDA::Compiler.new.compile_with_cache(STRUCT_LAYOUT_CODE).get_function('get_struct_layout')
  kernel.launch([itemsize, sizes, offsets], grid: 1, block: 1)

  itemsize = itemsize.to_a[0]
  sizes = sizes.to_a
  offsets = offsets.to_a
  puts "Overall structure itemsize: #{itemsize} bytes"
  puts "Structure members itemsize: #{sizes.inspect}"
  puts "Structure members offsets: #{offsets.inspect}"

  # Second step: place each member at the offset the device reported. pack's
  # '@' moves to a byte offset and pads what it skips, which is what a numpy
  # structured dtype with explicit offsets does on the CuPy side.
  members = [
    ['l4', [0, 1, 2, 3]],       # int4 a
    ['c', [4]],                 # char b
    ['d2', [5.0, 6.0]],         # double c[2]
    ['s', [7]],                 # short1 d
    ['Q3', [8, 9, 10]]          # unsigned long long e[3]
  ]
  template = members.each_with_index.map { |(fmt, _), i| "@#{offsets[i]}#{fmt}" }.join + "@#{itemsize}"
  s = members.flat_map { |_, values| values }.pack(template)
  raise "packed #{s.bytesize} bytes for a #{itemsize} byte struct" unless s.bytesize == itemsize

  puts "Complex structure value:\n  #{members.flat_map { |_, values| values }.inspect}"

  # Setup test kernel
  n = 8
  out = Cumo::DFloat.new(n)
  kernel = Cumo::CUDA::Compiler.new.compile_with_cache(KERNEL_CODE).get_function('test_kernel')
  kernel.launch([s, out], grid: 1, block: n)

  # the sum of all members of our complex struct instance is 55.0
  expected = Cumo::DFloat.new(n).seq * 55.0

  assert_array_almost_equal(expected, out)
  puts 'Kernel output matches expected value.'
  0
end

exit(main) if __FILE__ == $PROGRAM_NAME
