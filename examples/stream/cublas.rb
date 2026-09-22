# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cublas.rb
require 'cumo/narray'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' unless Integer(actual.eq(desired).count_false).zero?
end

x = Cumo::DFloat[1, 2, 3]
y = Cumo::DFloat[[1], [2], [3]]
expected = x.dot(y)
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
z = stream.with { x.dot(y) }
stream.synchronize
assert_array_equal(z, expected)

stream = Cumo::CUDA::Stream.new
stream.use
z = x.dot(y)
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(z, expected)
