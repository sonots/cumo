# frozen_string_literal: true

require 'numo/narray'
require 'cumo/narray'

CODE = <<~CUDA
  __device__ double3 operator+(const double3& lhs, const double3& rhs) {
      return make_double3(lhs.x + rhs.x,
                          lhs.y + rhs.y,
                          lhs.z + rhs.z);
  }

  extern "C" __global__ void sum_kernel(const double3* lhs,
                                              double3  rhs,
                                              double3* out) {
    int i = threadIdx.x;
    out[i] = lhs[i] + rhs;
  }
CUDA

def main
  n = 8

  # The kernel computes out = lhs+rhs where lhs and rhs are double3 vectors.
  # lhs is an array of N such vectors and rhs is double3 kernel parameter.
  #
  # An array of structs is an array whose trailing axes are the struct's
  # fields, so the double3* parameter takes a [N, 3] Cumo::DFloat, and the
  # double3 passed by value is the packed bytes of its three fields.
  lhs = Cumo::DFloat.new(n, 3).rand
  rhs = Numo::DFloat.new(3).rand
  out = Cumo::DFloat.new(n, 3)

  kernel = Cumo::CUDA::Compiler.new.compile_with_cache(CODE).get_function('sum_kernel')
  args = [lhs, rhs.to_a.pack('d3'), out]
  kernel.launch(args, grid: 1, block: n)

  expected = lhs + Cumo::DFloat.cast(rhs)[:new, true]
  raise 'kernel output does not match' unless Integer(expected.eq(out).count_false).zero?

  puts 'Kernel output matches expected value.'
  0
end

exit(main) if __FILE__ == $PROGRAM_NAME
