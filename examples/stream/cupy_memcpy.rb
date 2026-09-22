# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cupy_memcpy.rb
require 'numo/narray'
require 'cumo/narray'

# Cumo has no pinned memory pool or pinned allocator to install, so a buffer is
# allocated once here and reused, which is what the pool is for in CuPy.
def pin_memory(array)
  Cumo::CUDA::PinnedMemory.from_binary(array.to_binary)
end

SIZE = 1024 * 1024
x_cpu_src = Numo::SFloat.new(SIZE).seq
x_gpu_src = Cumo::SFloat.new(SIZE).seq

# synchronous
stream = Cumo::CUDA::Stream.null
start = stream.record
x_gpu_dst = Cumo::SFloat.new(SIZE)
x_gpu_dst.set(x_cpu_src.to_binary)
x_cpu_dst = Numo::SFloat.from_binary(x_gpu_src.get, [SIZE])
finish = stream.record

finish.synchronize
puts 'Synchronous Device to Host / Host to Device (ms)'
puts Cumo::CUDA.get_elapsed_time(start, finish)

# asynchronous
x_gpu_dst = Cumo::SFloat.new(SIZE)

x_pinned_cpu_src = pin_memory(x_cpu_src)
x_pinned_cpu_dst = Cumo::CUDA::PinnedMemory.new(x_gpu_src.byte_size)

stream_htod = Cumo::CUDA::Stream.new
start = nil
finish = nil
stream_htod.with do
  start = stream_htod.record
  x_gpu_dst.set(x_pinned_cpu_src)
  stream_dtoh = Cumo::CUDA::Stream.new
  stream_dtoh.with do
    x_gpu_src.get(x_pinned_cpu_dst)
  end
  stream_dtoh.synchronize
  finish = stream_htod.record
end

finish.synchronize
puts 'Asynchronous Device to Host / Host to Device (ms)'
puts Cumo::CUDA.get_elapsed_time(start, finish)

x_cpu_dst = Numo::SFloat.from_binary(x_pinned_cpu_dst.read, [SIZE])
raise 'the copies disagree' unless x_cpu_dst.to_a == x_cpu_src.to_a
raise 'the host to device copy disagrees' unless Numo::SFloat.cast(x_gpu_dst).to_a == x_cpu_src.to_a
