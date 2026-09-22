# frozen_string_literal: true

require 'cumo/narray'

device = Cumo::CUDA::Device.new
# Cumo's memory pool is a single global one, so there is no allocator to install
# and no pool object to ask; Cumo::CUDA::MemoryPool answers for it.
Cumo::DFloat.srand(1)

n = 10
zs = []
map_streams = []
stop_events = []
reduce_stream = Cumo::CUDA::Stream.new
n.times { map_streams << Cumo::CUDA::Stream.new }

start_time = Process.clock_gettime(Process::CLOCK_MONOTONIC)

# Map
map_streams.each do |stream|
  stream.use
  x = Cumo::DFloat.new(1, 1024**2).rand_norm
  y = Cumo::DFloat.new(1024**2, 1).rand_norm
  z = x.dot(y)
  zs << z
  stop_events << stream.record
end
Cumo::CUDA::Stream.null.use

# Block the `reduce_stream` until all events occur. This does not block host.
# This is not required when reduction is performed in the default (Stream.null)
# stream unless streams are created with `non_blocking: true`.
n.times { |i| reduce_stream.wait_event(stop_events[i]) }

# Reduce
reduce_stream.use
z = zs.reduce(:+)
Cumo::CUDA::Stream.null.use

device.synchronize
elapsed_time = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start_time
puts "elapsed time #{elapsed_time}"
puts "total bytes #{Cumo::CUDA::MemoryPool.total_bytes}"

# Free all blocks in the memory pool of streams
map_streams.each { |stream| Cumo::CUDA::MemoryPool.free_all_blocks(stream.handle) }
puts "total bytes #{Cumo::CUDA::MemoryPool.total_bytes}"
puts "sum #{Float(z.sum)}"
