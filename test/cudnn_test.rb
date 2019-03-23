require_relative "test_helper"

class CudnnTest < Test::Unit::TestCase
  float_types = [
    Cumo::SFloat,
    Cumo::DFloat,
  ]

  if ENV['DTYPE']
    float_types.select!{|type| type.to_s.downcase.include?(ENV['DTYPE'].downcase) }
  end

  float_types.each do |dtype|
    test "x.conv(w, b) #{dtype}" do
      batch_size = 2
      in_channels = 3
      out_channels = 2
      in_dims = [10, 7]
      kernel_size = [2, 3]

      x_shape = [batch_size, in_channels].concat(in_dims)
      w_shape = [out_channels, in_channels].concat(kernel_size)
      b_shape = [out_channels]
      stride = [3, 2]
      pad = [2, 0]

      x = dtype.ones(*x_shape)
      w = dtype.ones(*w_shape)
      b = dtype.ones(*b_shape)
      y = x.conv(w, b: b, stride: stride, pad: pad)

      # puts y

      # stride = [1, 1]
      # pad = [0, 0]
    end
  end
end
