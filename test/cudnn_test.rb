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
    sub_test_case "conv_2d" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [10, 7]
        @kernel_size = [2, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@out_channels, @in_channels].concat(@kernel_size)
        @b_shape = [@out_channels]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @b = dtype.ones(*@b_shape) * 2
      end

      test "x.conv(w) #{dtype}" do
        y = @x.conv(@w)
        assert { y.shape == [@batch_size, @out_channels, 9, 5] }
        assert y.to_a.flatten.all? {|e| e.to_i == 18 }
      end

      test "x.conv(w, b) #{dtype}" do
        y = @x.conv(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 9, 5] }
        assert y.to_a.flatten.all? {|e| e.to_i == 20 }
      end

      test "x.conv(w, b, stride, pad) #{dtype}" do
        stride = [3, 2]
        pad = [2, 0]
        y = @x.conv(@w, b: @b, stride: stride, pad: pad)
        assert { y.shape == [@batch_size, @out_channels, 5, 3] }
        assert y.to_a.flatten.all? {|e| e.to_i == 20 || e.to_i == 2 }
      end
    end

    sub_test_case "conv_nd" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [4, 3, 2]
        @kernel_size = [2, 3, 1]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@out_channels, @in_channels].concat(@kernel_size)
        @b_shape = [@out_channels]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @b = dtype.ones(*@b_shape) * 2
      end

      test "x.conv(w) #{dtype}" do
        y = @x.conv(@w)
        assert { y.shape == [@batch_size, @out_channels, 3, 1, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 18 }
      end

      test "x.conv(w, b) #{dtype}" do
        y = @x.conv(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 3, 1, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 20 }
      end

      test "x.conv(w, b, stride, pad) #{dtype}" do
        stride = [3, 2, 1]
        pad = [2, 1, 0]
        y = @x.conv(@w, b: @b, stride: stride, pad: pad)
        assert { y.shape == [@batch_size, @out_channels, 3, 2, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 14 || e.to_i == 2 }
      end
    end
  end
end
