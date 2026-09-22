# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cupy_event.rb
require 'cumo/narray'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' unless Integer(actual.eq(desired).count_false).zero?
end

START_EVENT = Cumo::CUDA::Event.new
STOP_EVENT = Cumo::CUDA::Event.new

def norm_with_elapsed_time(x)
  START_EVENT.record
  y = Cumo::NMath.sqrt((x * x).sum)
  STOP_EVENT.record
  STOP_EVENT.synchronize
  puts Cumo::CUDA.get_elapsed_time(START_EVENT, STOP_EVENT)
  y
end

x = Cumo::DFloat[1, 2, 3]

expected = norm_with_elapsed_time(x)
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
y = stream.with { norm_with_elapsed_time(x) }
stream.synchronize
assert_array_equal(y, expected)

stream = Cumo::CUDA::Stream.new
stream.use
y = norm_with_elapsed_time(x)
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(y, expected)
