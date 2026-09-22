# frozen_string_literal: true

require 'cumo/narray'

reduction = Cumo::CUDA::Compiler.new.compile_with_cache(<<~CUDA).get_function('reduction')
  extern "C" __global__ void reduction(const float* x, float* y, unsigned int size) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ntid = blockDim.x * gridDim.x;

    float value = 0;
    for (unsigned int i = tid; i < size; i += ntid) {
      value += x[i];
    }

    __shared__ float smem[1024];
    smem[threadIdx.x] = value;

    __syncthreads();

    if (threadIdx.x == 0) {
      value = 0;
      for (unsigned int i = 0; i < blockDim.x; i++) {
        value += smem[i];
      }
      atomicAdd(y, value);
    }
  }
CUDA

size = 2**22
x = Cumo::SFloat.new(size).rand_norm
# The CuPy original allocates this with cupy.empty. atomicAdd accumulates into
# it, so it has to start at zero.
y = Cumo::SFloat.zeros(1)

reduction.launch([x, y, [size].pack('L')], grid: 64, block: 1024)

puts Float(y[0])
puts Float(x.sum)
