require_relative "test_helper"

class CUDNNTest < Test::Unit::TestCase
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
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride=int, pad=int) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: 2, pad: 2)
        assert { y.shape == [@batch_size, @out_channels, 7, 5] }
        assert y.to_a.flatten.all? {|e| [20,2,8].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride=array, pad=array) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: [3, 2], pad: [2, 0])
        assert { y.shape == [@batch_size, @out_channels, 5, 3] }
        assert y.to_a.flatten.all? {|e| e.to_i == 20 || e.to_i == 2 }
        assert { @b.shape == @b_shape }
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
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b) #{dtype}" do
        y = @x.conv(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 3, 1, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 20 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride, pad) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: [3, 2, 1], pad: [2, 1, 0])
        assert { y.shape == [@batch_size, @out_channels, 3, 2, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 14 || e.to_i == 2 }
        assert { @b.shape == @b_shape }
      end
    end

    sub_test_case "conv_transpose_2d" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [5, 3]
        @kernel_size = [2, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@in_channels, @out_channels].concat(@kernel_size)
        @b_shape = [@out_channels]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @b = dtype.ones(*@b_shape) * 2
      end

      test "x.conv_transpose(w) #{dtype}" do
        y = @x.conv_transpose(@w)
        assert { y.shape == [@batch_size, @out_channels, 6, 5] }
      end

      test "x.conv_transpose(w, b) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 6, 5] }
        y_no_bias = @x.conv_transpose(@w)
        assert { y == y_no_bias + 2 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv_transpose(w, b, stride=int, pad=int) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b, stride: 2, pad: 2)
        assert { y.shape == [@batch_size, @out_channels, 6, 3] }
        assert y.to_a.flatten.all? {|e| e.to_i == 8 || e.to_i == 5 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv_transpose(w, b, stride=array, pad=array) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b, stride: [3, 2], pad: [2, 0])
        assert { y.shape == [@batch_size, @out_channels, 10, 7] }
        assert y.to_a.flatten.all? {|e| [8,5,2].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end
    end

    sub_test_case "conv_transpose_nd" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [4, 3, 2]
        @kernel_size = [2, 3, 1]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@in_channels, @out_channels].concat(@kernel_size)
        @b_shape = [@out_channels]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @b = dtype.ones(*@b_shape) * 2
      end

      test "x.conv_transpose(w) #{dtype}" do
        y = @x.conv_transpose(@w)
        assert { y.shape == [@batch_size, @out_channels, 5, 5, 2] }
        assert y.to_a.flatten.all? {|e| [3,6,9,12,18].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end

      test "x.conv_transpose(w, b) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 5, 5, 2] }
        y_no_bias = @x.conv_transpose(@w)
        assert { y == y_no_bias + 2 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv_transpose(w, b, stride, pad) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b, stride: [3, 2, 1], pad: [2, 1, 0])
        assert { y.shape == [@batch_size, @out_channels, 7, 5, 2] }
        assert y.to_a.flatten.all? {|e| [2,5,8].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end
    end

    sub_test_case "conv_backward_filter_2d" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [10, 7]
        @kernel_size = [2, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@out_channels, @in_channels].concat(@kernel_size)
        @y_shape = [@batch_size, @out_channels, 9, 5]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @dy = dtype.ones(*@y_shape)
      end

      test "x.conv_backward_filter(w) #{dtype}" do
        dw = @x.conv_backward_filter(@dy, @w_shape)
        assert { dw.shape == @w_shape }
        # TODO: assert values
      end
    end

    sub_test_case "conv_backward_filter_nd" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [4, 3, 2]
        @kernel_size = [2, 3, 1]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@out_channels, @in_channels].concat(@kernel_size)
        @y_shape = [@batch_size, @out_channels, 3, 1, 2]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @dy = dtype.ones(*@y_shape)
      end

      test "x.conv_backward_filter(w) #{dtype}" do
        dw = @x.conv_backward_filter(@dy, @w_shape)
        assert { dw.shape == @w_shape }
        # TODO: assert values
      end
    end

    sub_test_case "conv_backward_filter_2d" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @out_channels = 2
        @in_dims = [10, 7]
        @kernel_size = [2, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @w_shape = [@out_channels, @in_channels].concat(@kernel_size)
        @y_shape = [@batch_size, @out_channels, 9, 5]
        @x = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
        @dy = dtype.ones(*@y_shape)
      end

      test "x.conv_backward_filter(w) #{dtype}" do
        dw = @x.conv_backward_filter(@dy, @w_shape)
        assert { dw.shape == @w_shape }
        # TODO: assert values
      end
    end

    sub_test_case "batch_norm" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @reduced_shape = [1].concat(@x_shape[1..-1])
        @x = dtype.ones(*@x_shape) * 3
        @gamma = dtype.ones(*@reduced_shape) * 2
        @beta = dtype.ones(*@reduced_shape)
      end

      test "x.batch_norm(gamma, beta) #{dtype}" do
        y = @x.batch_norm(@gamma, @beta)
        assert { y.shape == @x_shape }
        assert { y == dtype.ones(*@x_shape) }
      end

      test "x.batch_norm(gamma, beta, axis: [0]) #{dtype}" do
        assert { @x.batch_norm(@gamma, @beta) == @x.batch_norm(@gamma, @beta, axis: [0]) }
      end

      test "x.batch_norm(gamma, beta, axis: [0, 2, 3]) #{dtype}" do
        reduced_shape = [1, @x_shape[1], 1, 1]
        gamma = dtype.ones(reduced_shape) * 2
        beta = dtype.ones(reduced_shape)
        y = @x.batch_norm(gamma, beta, axis: [0, 2, 3])
        assert { y.shape == @x_shape }
      end

      test "x.batch_norm(gamma, beta, running_mean, running_var) #{dtype}" do
        running_mean = dtype.ones(*@reduced_shape)
        running_var = dtype.ones(*@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, running_mean: running_mean, running_var: running_var)
        assert { y.shape == @x_shape }
        assert { y == dtype.ones(*@x_shape) }
      end

      test "x.batch_norm(gamma, beta, mean, inv_std) #{dtype}" do
        mean = dtype.new(*@reduced_shape)
        inv_std = dtype.new(*@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, mean: mean, inv_std: inv_std)
        assert { y.shape == @x_shape }
        assert { mean.shape == @reduced_shape }
        assert { inv_std.shape == @reduced_shape }
      end
    end

    sub_test_case "batch_norm_backward" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @reduced_shape = [1].concat(@x_shape[1..-1])
        @x = dtype.ones(*@x_shape) * 3
        @gamma = dtype.ones(*@reduced_shape) * 2
        @beta = dtype.ones(*@reduced_shape)
        @gy = dtype.ones(*@x_shape)
      end

      test "x.batch_norm_backward(gamma, gy) #{dtype}" do
        y = @x.batch_norm(@gamma, @beta)
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy)
        assert { gx.shape== @x_shape }
        assert { ggamma.shape== @reduced_shape }
        assert { gbeta.shape== @reduced_shape }
      end

      test "x.batch_norm_backward(gamma, gy, axis: [0,2,3]) #{dtype}" do
        @reduced_shape = [1, @x_shape[1], 1, 1]
        @gamma = dtype.ones(@reduced_shape) * 2
        @beta = dtype.ones(@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, axis: [0,2,3])
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy, axis: [0,2,3])
        assert { gx.shape== @x_shape }
        assert { ggamma.shape== @reduced_shape }
        assert { gbeta.shape== @reduced_shape }
      end

      test "x.batch_norm_backward(gamma, gy, mean:, inv_std:) #{dtype}" do
        mean = dtype.new(*@reduced_shape)
        inv_std = dtype.new(*@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, mean: mean, inv_std: inv_std)
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy, mean: mean, inv_std: inv_std)
        assert { gx.shape== @x_shape }
        assert { ggamma.shape== @reduced_shape }
        assert { gbeta.shape== @reduced_shape }
      end

    end

    sub_test_case "max_pool" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @ksize = [3] * @in_dims.size
        @x = dtype.ones(*@x_shape) * 3
      end

      test "x.max_pool(ksize) #{dtype}" do
        y = @x.max_pool(@ksize)
        assert { y.shape == [@batch_size, @in_channels, 1, 1] }
        assert y.to_a.flatten.all? {|e| e.to_i == 3 }
      end

      test "x.max_pool(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y = @x.max_pool(@ksize, stride: stride, pad: pad)
        assert { y.shape == [@batch_size, @in_channels, 3, 2] }
        assert y.to_a.flatten.all? {|e| e.to_i == 3 }
      end
    end

    sub_test_case "avg_pool(pad_value: nil)" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @ksize = [3] * @in_dims.size
        @x = dtype.ones(*@x_shape) * 3
      end

      test "x.avg_pool(ksize) #{dtype}" do
        y = @x.avg_pool(@ksize)
        assert { y.shape == [@batch_size, @in_channels, 1, 1] }
        assert y.to_a.flatten.all? {|e| e.to_i == 3 }
      end

      test "x.avg_pool(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y = @x.avg_pool(@ksize, stride: stride, pad: pad)
        assert { y.shape == [@batch_size, @in_channels, 3, 2] }
        # TODO: assert values
      end
    end

    sub_test_case "avg_pool(pad_value: 0)" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @ksize = [3] * @in_dims.size
        @x = dtype.ones(*@x_shape) * 3
      end

      test "x.avg_pool(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y_pad_0 = @x.avg_pool(@ksize, pad_value: 0, stride: stride, pad: pad)
        y_pad_nil = @x.avg_pool(@ksize, pad_value: nil, stride: stride, pad: pad)
        assert { y_pad_0.shape == y_pad_nil.shape }
        assert { y_pad_0 != y_pad_nil }
      end
    end

    sub_test_case "max_pool_backward" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @ksize = [3] * @in_dims.size
        @x = dtype.ones(*@x_shape) * 3
      end

      test "x.max_pool_backward(ksize) #{dtype}" do
        y = @x.max_pool(@ksize)
        gy = dtype.ones(*y.shape)
        gx = @x.max_pool_backward(y, gy, @ksize)
        assert { gx.shape == @x.shape }
        # TODO: assert values
      end

      test "x.max_pool_backward(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y = @x.max_pool(@ksize, stride: stride, pad: pad)
        gy = dtype.ones(*y.shape)
        gx = @x.max_pool_backward(y, gy, @ksize, stride: stride, pad: pad)
        assert { gx.shape == @x.shape }
        # TODO: assert values
      end
    end

    sub_test_case "avg_pool_backward" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @ksize = [3] * @in_dims.size
        @x = dtype.ones(*@x_shape) * 3
      end

      test "x.avg_pool_backward(ksize) #{dtype}" do
        y = @x.avg_pool(@ksize)
        gy = dtype.ones(*y.shape)
        gx = @x.avg_pool_backward(y, gy, @ksize)
        assert { gx.shape == @x.shape }
        # TODO: assert values
      end

      test "x.avg_pool_backward(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y = @x.avg_pool(@ksize, stride: stride, pad: pad)
        gy = dtype.ones(*y.shape)
        gx = @x.avg_pool_backward(y, gy, @ksize, stride: stride, pad: pad)
        assert { gx.shape == @x.shape }
        # TODO: assert values
      end
    end
  end
end
