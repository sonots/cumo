# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/cufft.rb
require 'cumo/fft'

def assert_array_equal(actual, desired)
  raise 'arrays are not equal' if Integer(actual.eq(desired).count_false).nonzero?
end

x = Cumo::DFloat[1, 0, 3, 0, 5, 0, 7, 0, 9]
expected_f = Cumo::FFT.fft(x)
Cumo::CUDA::Device.new.synchronize

stream = Cumo::CUDA::Stream.new
f = stream.with { Cumo::FFT.fft(x) }
stream.synchronize
assert_array_equal(f, expected_f)

stream = Cumo::CUDA::Stream.new
stream.use
f = Cumo::FFT.fft(x)
stream.synchronize
Cumo::CUDA::Stream.null.use
assert_array_equal(f, expected_f)
