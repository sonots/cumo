# frozen_string_literal: true

require 'numo/narray'
require 'cumo/narray'

CODE = <<~CUDA
  template<typename T>
  struct Matrix {
      T value[4][4];

      __device__ T& operator() (int i, int j) {
          return this->value[i][j];
      }

      __device__ const T& operator() (int i, int j) const {
          return this->value[i][j];
      }
  };

  template<typename T>
  __device__ Matrix<T> operator+ (const Matrix<T>& lhs, const Matrix<T>& rhs) {
      Matrix<T> res;
      for (int i = 0; i<4; i++) {
          for (int j = 0; j<4; j++) {
              res(i,j) = lhs(i,j) + rhs(i,j);
          }
      }
      return res;
  }

  template<typename T>
  __device__ Matrix<T> operator* (const Matrix<T>& lhs, const Matrix<T>& rhs) {
      Matrix<T> res;
      for (int i = 0; i<4; i++) {
          for (int j = 0; j<4; j++) {
              res(i,j) = T(0);
              for (int k = 0; k<4; k++) {
                  res(i,j) += lhs(i,k) * rhs(k,j);
              }
          }
      }
      return res;
  }

  template<typename T>
  __global__ void kernel(const Matrix<T>* A,
                         const Matrix<T>* B,
                         const Matrix<T>  C,
                               Matrix<T>* out) {
    int i = threadIdx.x;
    out[i] = A[i] * B[i] + C;
  }
CUDA

def assert_array_almost_equal(actual, desired, decimal: 6)
  diff = (Cumo::DFloat.cast(actual) - Cumo::DFloat.cast(desired)).abs.max
  return if Float(diff) < 1.5 * (10**-decimal)

  raise "Arrays are not almost equal to #{decimal} decimals: largest difference #{Float(diff)}"
end

def main
  n = 8
  module_ = Cumo::CUDA::Compiler.new.compile_with_cache(
    CODE, name_expressions: ['kernel<float>', 'kernel<double>']
  )

  # The kernel computes out = A*B+C where A, B and C are 4x4 matrices.
  # A and B are arrays of N such matrices and C is a matrix kernel parameter.
  [['float', Cumo::SFloat, Numo::SFloat, 'f*'], ['double', Cumo::DFloat, Numo::DFloat, 'd*']]
    .each do |ctype, xf, hf, pack|
    a = xf.new(n, 4, 4).rand
    b = xf.new(n, 4, 4).rand
    c = hf.new(4, 4).rand
    out = xf.new(n, 4, 4)

    kernel = module_.get_function("kernel<#{ctype}>")
    args = [a, b, c.flatten.to_a.pack(pack), out]
    kernel.launch(args, grid: 1, block: n)

    expected = a.dot(b) + xf.cast(c)[:new, true, true]

    assert_array_almost_equal(expected, out, decimal: ctype == 'float' ? 5 : 12)
    puts "Kernel output matches expected value for type '#{ctype}'."
  end
  0
end

exit(main) if __FILE__ == $PROGRAM_NAME
