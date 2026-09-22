# frozen_string_literal: true

require 'cumo/narray'

reduction = Cumo::CUDA::Compiler.new.compile_with_cache(<<~CUDA).get_function('reduction')
  extern "C" __global__ void reduction(const float* x, float* y, unsigned int size) {
    unsigned int tid = threadIdx.x;
    unsigned int ntid = blockDim.x;

    float value = 0;
    for (unsigned int i = tid; i < size; i += ntid) {
      value += x[i];
    }

    __shared__ float smem[1024];
    smem[tid] = value;

    __syncthreads();

    if (tid == 0) {
      value = 0;
      for (unsigned int i = 0; i < ntid; i++) {
        value += smem[i];
      }
      y[0] = value;
    }
  }
CUDA

size = 2**22
x = Cumo::SFloat.new(size).rand_norm
y = Cumo::SFloat.new(1)

reduction.launch([x, y, [size].pack('L')], grid: 1, block: 1024)

puts Float(y[0])
puts Float(x.sum)
