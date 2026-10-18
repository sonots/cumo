# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cusolver.rb
require 'cumo/linalg'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' if Integer(actual.eq(desired).count_false).nonzero?
end

x = Cumo::DFloat[[1, 0, 3], [0, 5, 0], [7, 0, 9]]
expected_w, expected_v = Cumo::Linalg.eigh(x, uplo: 'U')
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
w, v = stream.with { Cumo::Linalg.eigh(x, uplo: 'U') }
stream.synchronize
assert_array_equal(w, expected_w)
assert_array_equal(v, expected_v)

stream = Cumo::CUDA::Stream.new
stream.use
w, v = Cumo::Linalg.eigh(x, uplo: 'U')
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(w, expected_w)
assert_array_equal(v, expected_v)
