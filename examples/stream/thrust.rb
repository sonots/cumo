# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/thrust.rb
require 'cumo/narray'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' unless Integer(actual.eq(desired).count_false).zero?
end

x = Cumo::DFloat[1, 3, 2]
expected = x.sort
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
y = stream.with { x.sort }
stream.synchronize
assert_array_equal(y, expected)

stream = Cumo::CUDA::Stream.new
stream.use
y = x.sort
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(y, expected)
