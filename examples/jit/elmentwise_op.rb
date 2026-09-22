# frozen_string_literal: true

require 'cumo/narray'

elementwise_copy = Cumo::CUDA::Compiler.new.compile_with_cache(<<~CUDA).get_function('elementwise_copy')
  extern "C" __global__ void elementwise_copy(const float* x, float* y, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ntid = gridDim.x * blockDim.x;
    for (unsigned int i = tid; i < size; i += ntid) {
      y[i] = x[i];
    }
  }
CUDA

size = 2**22
x = Cumo::SFloat.new(size).rand_norm
y = Cumo::SFloat.new(size)

elementwise_copy.launch([x, y, [size].pack('L')], grid: 128, block: 1024)

raise 'x and y differ' unless Integer(x.eq(y).count_true) == size
