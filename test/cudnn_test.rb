# frozen_string_literal: true

require_relative "test_helper"

class CUDNNTest < Test::Unit::TestCase
  float_types = [
    Cumo::SFloat,
    Cumo::DFloat,
  ]

  if ENV['DTYPE']
    float_types.select! { |type| type.to_s.downcase.include?(ENV['DTYPE'].downcase) }
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
        assert y.to_a.flatten.all? { |e| e.to_i == 18 }
      end

      test "x.conv(w, b) #{dtype}" do
        y = @x.conv(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 9, 5] }
        assert y.to_a.flatten.all? { |e| e.to_i == 20 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride=int, pad=int) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: 2, pad: 2)
        assert { y.shape == [@batch_size, @out_channels, 7, 5] }
        assert y.to_a.flatten.all? { |e| [20, 2, 8].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride=array, pad=array) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: [3, 2], pad: [2, 0])
        assert { y.shape == [@batch_size, @out_channels, 5, 3] }
        assert y.to_a.flatten.all? { |e| e.to_i == 20 || e.to_i == 2 }
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
        assert y.to_a.flatten.all? { |e| e.to_i == 18 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b) #{dtype}" do
        y = @x.conv(@w, b: @b)
        assert { y.shape == [@batch_size, @out_channels, 3, 1, 2] }
        assert y.to_a.flatten.all? { |e| e.to_i == 20 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv(w, b, stride, pad) #{dtype}" do
        y = @x.conv(@w, b: @b, stride: [3, 2, 1], pad: [2, 1, 0])
        assert { y.shape == [@batch_size, @out_channels, 3, 2, 2] }
        assert y.to_a.flatten.all? { |e| e.to_i == 14 || e.to_i == 2 }
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
        assert y.to_a.flatten.all? { |e| e.to_i == 8 || e.to_i == 5 }
        assert { @b.shape == @b_shape }
      end

      test "x.conv_transpose(w, b, stride=array, pad=array) #{dtype}" do
        y = @x.conv_transpose(@w, b: @b, stride: [3, 2], pad: [2, 0])
        assert { y.shape == [@batch_size, @out_channels, 10, 7] }
        assert y.to_a.flatten.all? { |e| [8, 5, 2].include?(e.to_i) }
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
        assert y.to_a.flatten.all? { |e| [3, 6, 9, 12, 18].include?(e.to_i) }
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
        assert y.to_a.flatten.all? { |e| [2, 5, 8].include?(e.to_i) }
        assert { @b.shape == @b_shape }
      end
    end

    sub_test_case "conv_grad_w_2d" do
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

      test "x.conv_grad_w(w) #{dtype}" do
        dw = @x.conv_grad_w(@dy, @w_shape)
        assert { dw.shape == @w_shape }
        # TODO: assert values
      end
    end

    sub_test_case "conv_grad_w_nd" do
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

      test "x.conv_grad_w(w) #{dtype}" do
        dw = @x.conv_grad_w(@dy, @w_shape)
        assert { dw.shape == @w_shape }
        # TODO: assert values
      end
    end

    sub_test_case "conv_grad_w_2d" do
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

      test "x.conv_grad_w(w) #{dtype}" do
        dw = @x.conv_grad_w(@dy, @w_shape)
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
        assert_in_delta(y, dtype.ones(*@x_shape), 1e-3)
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
        assert_in_delta(y, dtype.ones(*@x_shape), 1e-3)
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

    sub_test_case "batch_norm with an axis that folds the channel away" do
      setup do
        @x = dtype.new(8, 256).seq
        @one = dtype.ones(1)
        @zero = dtype.zeros(1)
      end

      # cuDNN derives its parameter descriptor from x with a SPATIAL mode and
      # reaches 256 channels in each buffer, whatever the axis reduced them to
      test "batch_norm rejects it #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm(@one, @zero, axis: [0, 1]) }
        assert_raise(Cumo::NArray::ShapeError) do
          @x.batch_norm(@one, @zero, axis: [0, 1], running_mean: @zero, running_var: @one)
        end
        assert_raise(Cumo::NArray::ShapeError) do
          @x.batch_norm(@one, @zero, axis: [0, 1], mean: @zero, inv_std: @zero)
        end
      end

      test "fixed_batch_norm and batch_norm_backward reject it #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) do
          @x.fixed_batch_norm(@one, @zero, @zero, @one, axis: [0, 1])
        end
        assert_raise(Cumo::NArray::ShapeError) do
          @x.batch_norm_backward(@one, dtype.ones(8, 256), axis: [0, 1])
        end
      end

      test "the same shapes with the default axis still go through #{dtype}" do
        gamma = dtype.ones(1, 256)
        beta = dtype.zeros(1, 256)
        assert { @x.batch_norm(gamma, beta).shape == [8, 256] }
        assert { @x.fixed_batch_norm(gamma, beta, beta, gamma).shape == [8, 256] }
        assert { @x.batch_norm_backward(gamma, dtype.ones(8, 256)).first.shape == [8, 256] }
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
        @x.batch_norm(@gamma, @beta)
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy)
        assert { gx.shape == @x_shape }
        assert { ggamma.shape == @reduced_shape }
        assert { gbeta.shape == @reduced_shape }
      end

      test "x.batch_norm_backward(gamma, gy, axis: [0,2,3]) #{dtype}" do
        @reduced_shape = [1, @x_shape[1], 1, 1]
        @gamma = dtype.ones(@reduced_shape) * 2
        @beta = dtype.ones(@reduced_shape)
        @x.batch_norm(@gamma, @beta, axis: [0, 2, 3])
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy, axis: [0, 2, 3])
        assert { gx.shape == @x_shape }
        assert { ggamma.shape == @reduced_shape }
        assert { gbeta.shape == @reduced_shape }
      end

      test "x.batch_norm_backward(gamma, gy, mean:, inv_std:) #{dtype}" do
        mean = dtype.new(*@reduced_shape)
        inv_std = dtype.new(*@reduced_shape)
        @x.batch_norm(@gamma, @beta, mean: mean, inv_std: inv_std)
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy, mean: mean, inv_std: inv_std)
        assert { gx.shape == @x_shape }
        assert { ggamma.shape == @reduced_shape }
        assert { gbeta.shape == @reduced_shape }
      end
    end

    sub_test_case "fixed_batch_norm" do
      setup do
        @batch_size = 2
        @in_channels = 3
        @in_dims = [5, 3]
        @x_shape = [@batch_size, @in_channels].concat(@in_dims)
        @reduced_shape = [1].concat(@x_shape[1..-1])
        @x = dtype.ones(*@x_shape) * 3
        @gamma = dtype.ones(*@reduced_shape) * 2
        @beta = dtype.ones(*@reduced_shape)
        @mean = dtype.ones(*@reduced_shape)
        @var = dtype.ones(*@reduced_shape)
      end

      test "x.fixed_batch_norm(gamma, beta, mean, var) #{dtype}" do
        y = @x.fixed_batch_norm(@gamma, @beta, @mean, @var)
        assert { y.shape == @x_shape }
        # TODO: check output values
      end

      test "x.fixed_batch_norm(gamma, beta, mean, var, axis: [0]) #{dtype}" do
        assert { @x.fixed_batch_norm(@gamma, @beta, @mean, @var) == @x.fixed_batch_norm(@gamma, @beta, @mean, @var, axis: [0]) }
      end

      test "x.fixed_batch_norm(gamma, beta, mean, var, axis: [0, 2, 3]) #{dtype}" do
        reduced_shape = [1, @x_shape[1], 1, 1]
        gamma = dtype.ones(reduced_shape) * 2
        beta = dtype.ones(reduced_shape)
        mean = dtype.ones(reduced_shape) * 2
        var = dtype.ones(reduced_shape)
        y = @x.fixed_batch_norm(gamma, beta, mean, var, axis: [0, 2, 3])
        assert { y.shape == @x_shape }
        # TODO: check output values
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
        assert y.to_a.flatten.all? { |e| e.to_i == 3 }
      end

      test "x.max_pool(ksize, stride:, pad:) #{dtype}" do
        stride = [2] * @in_dims.size
        pad = [1] * @in_dims.size
        y = @x.max_pool(@ksize, stride: stride, pad: pad)
        assert { y.shape == [@batch_size, @in_channels, 3, 2] }
        assert y.to_a.flatten.all? { |e| e.to_i == 3 }
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
        assert y.to_a.flatten.all? { |e| e.to_i == 3 }
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

    sub_test_case "conv_grad_w gy shape" do
      setup do
        @x = dtype.ones(2, 3, 10, 7)
        @w_shape = [2, 3, 2, 3]
      end

      test "the spatial size of gy has to be the convolution output size #{dtype}" do
        assert { @x.conv_grad_w(dtype.ones(2, 2, 9, 5), @w_shape).shape == @w_shape }
        [[2, 2, 8, 5], [2, 2, 12, 5], [2, 2, 9, 4]].each do |shape|
          e = assert_raise(Cumo::NArray::ShapeError) { @x.conv_grad_w(dtype.ones(*shape), @w_shape) }
          assert { e.message.include?("does not match with the convolution output size") }
        end
      end

      test "a stride that does not divide evenly takes the floor #{dtype}" do
        x = dtype.ones(2, 3, 10, 10)
        w_shape = [2, 3, 2, 2]
        assert { x.conv_grad_w(dtype.ones(2, 2, 3, 3), w_shape, stride: 3).shape == w_shape }
        assert_raise(Cumo::NArray::ShapeError) { x.conv_grad_w(dtype.ones(2, 2, 4, 4), w_shape, stride: 3) }
      end
    end

    sub_test_case "given output arrays" do
      setup do
        @x_shape = [2, 3, 5, 3]
        @reduced_shape = [1].concat(@x_shape[1..-1])
        @w_shape = [2, 3, 2, 3]
        @ksize = [3, 3]
        @x = dtype.ones(*@x_shape) * 3
        @gamma = dtype.ones(*@reduced_shape) * 2
        @beta = dtype.ones(*@reduced_shape)
        @mean = dtype.ones(*@reduced_shape)
        @var = dtype.ones(*@reduced_shape)
        @gy = dtype.ones(*@x_shape)
        @w = dtype.ones(*@w_shape)
      end

      test "batch_norm(y:) #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm(@gamma, @beta, y: dtype.zeros(1)) }
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm(@gamma, @beta, y: dtype.zeros(2, 3, 5, 6)[true, true, true, 0..2]) }
        y = dtype.zeros(*@x_shape)
        assert { @x.batch_norm(@gamma, @beta, y: y).equal?(y) }
        assert { y == @x.batch_norm(@gamma, @beta) }
      end

      test "fixed_batch_norm(y:) #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) { @x.fixed_batch_norm(@gamma, @beta, @mean, @var, y: dtype.zeros(1)) }
        y = dtype.zeros(*@x_shape)
        assert { y == @x.fixed_batch_norm(@gamma, @beta, @mean, @var, y: y) }
      end

      test "batch_norm_backward(gx:, ggamma:, gbeta:) #{dtype}" do
        @x.batch_norm(@gamma, @beta)
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm_backward(@gamma, @gy, gx: dtype.zeros(1)) }
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm_backward(@gamma, @gy, ggamma: dtype.zeros(1)) }
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm_backward(@gamma, @gy, gbeta: dtype.zeros(1)) }
        gx = dtype.zeros(*@x_shape)
        ggamma = dtype.zeros(*@reduced_shape)
        gbeta = dtype.zeros(*@reduced_shape)
        assert { @x.batch_norm_backward(@gamma, @gy, gx: gx, ggamma: ggamma, gbeta: gbeta) == [gx, ggamma, gbeta] }
      end

      test "max_pool(y:) and max_pool_backward(gx:) #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) { @x.max_pool(@ksize, y: dtype.zeros(1)) }
        y = @x.max_pool(@ksize)
        gy = dtype.ones(*y.shape)
        assert_raise(Cumo::NArray::ShapeError) { @x.max_pool_backward(y, gy, @ksize, gx: dtype.zeros(1)) }
        gx = dtype.zeros(*@x_shape)
        assert { @x.max_pool_backward(y, gy, @ksize, gx: gx).equal?(gx) }
      end

      test "max_pool_backward checks y and gy #{dtype}" do
        # both are read through a descriptor built from y's shape and this
        # class's dtype, so cuDNN never learns how long either one is
        other = dtype == Cumo::SFloat ? Cumo::DFloat : Cumo::SFloat
        y = @x.max_pool(@ksize)
        gy = dtype.ones(*y.shape)
        assert_raise(Cumo::NArray::ShapeError) { @x.max_pool_backward(y, dtype.ones(1), @ksize) }
        assert_raise(Cumo::NArray::ShapeError) { @x.max_pool_backward(dtype.ones(1), gy, @ksize) }
        assert_raise(Cumo::NArray::ShapeError) { @x.max_pool_backward(dtype.ones(*@x_shape), dtype.ones(*@x_shape), @ksize) }
        assert_raise(TypeError) { @x.max_pool_backward(y, other.ones(*y.shape), @ksize) }
        assert_raise(TypeError) { @x.max_pool_backward(other.ones(*y.shape), gy, @ksize) }
        assert_raise(Cumo::NArray::ShapeError) { @x.avg_pool_backward(y, dtype.ones(1), @ksize) }
        # a y and gy of the pooling output shape still go through
        assert { @x.max_pool_backward(y, gy, @ksize).sum.to_f == y.size }
      end

      test "conv(y:), conv_transpose(y:) and conv_grad_w(gw:) #{dtype}" do
        assert_raise(Cumo::NArray::ShapeError) { @x.conv(@w, y: dtype.zeros(1)) }
        y = @x.conv(@w)
        assert_raise(Cumo::NArray::ShapeError) { @x.conv_grad_w(y, @w_shape, gw: dtype.zeros(1)) }
        gw = dtype.zeros(*@w_shape)
        assert { @x.conv_grad_w(y, @w_shape, gw: gw).equal?(gw) }
        assert_raise(Cumo::NArray::ShapeError) { @x.conv_transpose(@w, y: dtype.zeros(1)) }
      end
    end
  end
end
