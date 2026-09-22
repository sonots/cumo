# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cupy_kernel.rb
require 'cumo/narray'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' unless Integer(actual.eq(desired).count_false).zero?
end

def norm(x)
  Cumo::NMath.sqrt((x * x).sum)
end

x = Cumo::DFloat[1, 2, 3]
expected = norm(x)
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
y = stream.with { norm(x) }
stream.synchronize
assert_array_equal(y, expected)

stream = Cumo::CUDA::Stream.new
stream.use
y = norm(x)
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(y, expected)
