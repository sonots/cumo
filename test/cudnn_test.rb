# frozen_string_literal: true

require_relative "test_helper"

class CUDNNTest < Test::Unit::TestCase
  float_types = [
    Cumo::HFloat,
    Cumo::SFloat,
    Cumo::DFloat,
  ]

  if ENV['DTYPE']
    float_types.select! { |type| type.to_s.downcase.include?(ENV['DTYPE'].downcase) }
  end

  float_types.each do |dtype|
    # cuDNN derives the batch norm parameter descriptor from x and widens it to
    # float for a half x, so those arrays take this class rather than x's own
    param_type = dtype == Cumo::HFloat ? Cumo::SFloat : dtype

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
        @gamma = param_type.ones(*@reduced_shape) * 2
        @beta = param_type.ones(*@reduced_shape)
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
        gamma = param_type.ones(reduced_shape) * 2
        beta = param_type.ones(reduced_shape)
        y = @x.batch_norm(gamma, beta, axis: [0, 2, 3])
        assert { y.shape == @x_shape }
      end

      test "x.batch_norm(gamma, beta, running_mean, running_var) #{dtype}" do
        running_mean = param_type.ones(*@reduced_shape)
        running_var = param_type.ones(*@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, running_mean: running_mean, running_var: running_var)
        assert { y.shape == @x_shape }
        assert_in_delta(y, dtype.ones(*@x_shape), 1e-3)
      end

      test "x.batch_norm(gamma, beta, mean, inv_std) #{dtype}" do
        mean = param_type.new(*@reduced_shape)
        inv_std = param_type.new(*@reduced_shape)
        y = @x.batch_norm(@gamma, @beta, mean: mean, inv_std: inv_std)
        assert { y.shape == @x_shape }
        assert { mean.shape == @reduced_shape }
        assert { inv_std.shape == @reduced_shape }
      end
    end

    sub_test_case "batch_norm with an axis that folds the channel away" do
      setup do
        @x = dtype.new(8, 256).seq
        @one = param_type.ones(1)
        @zero = param_type.zeros(1)
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
        gamma = param_type.ones(1, 256)
        beta = param_type.zeros(1, 256)
        assert { @x.batch_norm(gamma, beta).shape == [8, 256] }
        assert { @x.fixed_batch_norm(gamma, beta, beta, gamma).shape == [8, 256] }
        assert { @x.batch_norm_backward(gamma, dtype.ones(8, 256)).first.shape == [8, 256] }
      end
    end

    # The mode cuDNN is given comes from x's shape, not from the axis, so an
    # axis that names nothing in x answers as though a different one had been
    # given rather than failing. The sizes below are the ones that reach the
    # kernel, so nothing else can raise first.
    # cuDNN derives the parameter descriptor from x, giving it x's own type
    # except for a half x, where it widens to float. What keeps the descriptor
    # and the buffer behind it in step is that every parameter is held to
    # exactly the class that descriptor carries.
    sub_test_case "a parameter of another dtype" do
      setup do
        @other = (param_type == Cumo::DFloat) ? Cumo::SFloat : Cumo::DFloat
        @x = dtype.new(2, 4, 3, 3).seq(1)
        @gamma = param_type.ones(4)
        @beta = param_type.zeros(4)
        @gy = dtype.ones(2, 4, 3, 3)
        @axis = [0, 2, 3]
      end

      test "batch_norm rejects it #{dtype}" do
        assert_raise(TypeError) { @x.batch_norm(@other.ones(4), @beta, axis: @axis) }
        assert_raise(TypeError) { @x.batch_norm(@gamma, @other.zeros(4), axis: @axis) }
        assert_raise(TypeError) do
          @x.batch_norm(@gamma, @beta, axis: @axis, running_mean: @other.zeros(4), running_var: @other.ones(4))
        end
        assert_raise(TypeError) do
          @x.batch_norm(@gamma, @beta, axis: @axis, mean: @other.zeros(4), inv_std: @other.zeros(4))
        end
      end

      test "fixed_batch_norm and batch_norm_backward reject it #{dtype}" do
        assert_raise(TypeError) do
          @x.fixed_batch_norm(@other.ones(4), @beta, @beta, @gamma, axis: @axis)
        end
        assert_raise(TypeError) do
          @x.fixed_batch_norm(@gamma, @beta, @other.zeros(4), @gamma, axis: @axis)
        end
        assert_raise(TypeError) { @x.batch_norm_backward(@other.ones(4), @gy, axis: @axis) }
        assert_raise(TypeError) { @x.batch_norm_backward(@gamma, @other.ones(2, 4, 3, 3), axis: @axis) }
      end

      test "the same call with x's own dtype goes through #{dtype}" do
        assert { @x.batch_norm(@gamma, @beta, axis: @axis).class == dtype }
        assert { @x.fixed_batch_norm(@gamma, @beta, @beta, @gamma, axis: @axis).class == dtype }
        assert { @x.batch_norm_backward(@gamma, @gy, axis: @axis).first.class == dtype }
      end
    end

    sub_test_case "an axis that does not name x's dimensions" do
      setup do
        @x = dtype.new(2, 4, 3, 3).seq(1)
        @gamma = param_type.ones(4)
        @beta = param_type.zeros(4)
        @gy = dtype.ones(2, 4, 3, 3)
      end

      test "batch_norm rejects an axis outside x #{dtype}" do
        assert_raise(ArgumentError) { @x.batch_norm(param_type.ones(72), param_type.zeros(72), axis: [5]) }
        assert_raise(ArgumentError) { @x.batch_norm(param_type.ones(72), param_type.zeros(72), axis: [-1]) }
      end

      test "batch_norm rejects a repeated or unordered axis #{dtype}" do
        assert_raise(ArgumentError) { @x.batch_norm(param_type.ones(36), param_type.zeros(36), axis: [0, 0]) }
        assert_raise(ArgumentError) { @x.batch_norm(@gamma, @beta, axis: [3, 0, 2]) }
        assert_raise(ArgumentError) { @x.batch_norm(@gamma, @beta, axis: [0, 3, 2]) }
      end

      test "fixed_batch_norm and batch_norm_backward reject it too #{dtype}" do
        assert_raise(ArgumentError) do
          @x.fixed_batch_norm(@gamma, @beta, @beta, @gamma, axis: [5])
        end
        assert_raise(ArgumentError) do
          @x.fixed_batch_norm(@gamma, @beta, @beta, @gamma, axis: [0, 0])
        end
        assert_raise(ArgumentError) { @x.batch_norm_backward(@gamma, @gy, axis: [-1]) }
        assert_raise(ArgumentError) { @x.batch_norm_backward(@gamma, @gy, axis: [3, 0, 2]) }
      end

      test "the spatial axis and the default still go through #{dtype}" do
        assert { @x.batch_norm(@gamma, @beta, axis: [0, 2, 3]).shape == [2, 4, 3, 3] }
        assert { @x.fixed_batch_norm(@gamma, @beta, @beta, @gamma, axis: [0, 2, 3]).shape == [2, 4, 3, 3] }
        assert { @x.batch_norm_backward(@gamma, @gy, axis: [0, 2, 3]).first.shape == [2, 4, 3, 3] }

        flat = dtype.new(8, 4).seq(1)
        assert { flat.batch_norm(@gamma, @beta).shape == [8, 4] }
        assert { flat.batch_norm(@gamma, @beta, axis: [0]).shape == [8, 4] }
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
        @gamma = param_type.ones(*@reduced_shape) * 2
        @beta = param_type.ones(*@reduced_shape)
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
        @gamma = param_type.ones(@reduced_shape) * 2
        @beta = param_type.ones(@reduced_shape)
        @x.batch_norm(@gamma, @beta, axis: [0, 2, 3])
        gx, ggamma, gbeta = @x.batch_norm_backward(@gamma, @gy, axis: [0, 2, 3])
        assert { gx.shape == @x_shape }
        assert { ggamma.shape == @reduced_shape }
        assert { gbeta.shape == @reduced_shape }
      end

      test "x.batch_norm_backward(gamma, gy, mean:, inv_std:) #{dtype}" do
        mean = param_type.new(*@reduced_shape)
        inv_std = param_type.new(*@reduced_shape)
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
        @gamma = param_type.ones(*@reduced_shape) * 2
        @beta = param_type.ones(*@reduced_shape)
        @mean = param_type.ones(*@reduced_shape)
        @var = param_type.ones(*@reduced_shape)
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
        gamma = param_type.ones(reduced_shape) * 2
        beta = param_type.ones(reduced_shape)
        mean = param_type.ones(reduced_shape) * 2
        var = param_type.ones(reduced_shape)
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

    sub_test_case "a non-contiguous input" do
      setup do
        @x = dtype.new(1, 1, 8, 8).seq
        @max = Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX
      end

      # y and gy are copied before they are read, so a strided one is allowed;
      # this pins that, not the copy itself. cuDNN recomputes the argmax from x
      # and never reads y's values, so no assertion here can see a wrong y.
      test "pooling_backward takes a strided y #{dtype}" do
        y = @x.pooling_forward(@max, 2)
        pad = dtype.zeros(1, 1, 4, 8)
        pad[true, true, true, 0...4] = y
        y_strided = pad[true, true, true, 0...4]
        assert { !y_strided.contiguous? }
        gy = dtype.new(1, 1, 4, 4).seq(1)
        gx = @x.pooling_backward(@max, y_strided, gy, 2)
        assert { gx == @x.pooling_backward(@max, y, gy, 2) }
        assert { gx.to_a.flatten.count { |v| v != 0 } == y.size }
      end

      test "pooling_backward takes a strided gy #{dtype}" do
        y = @x.pooling_forward(@max, 2)
        pad = dtype.zeros(1, 1, 4, 8)
        pad[true, true, true, 0...4] = dtype.new(1, 1, 4, 4).seq(1)
        gy_strided = pad[true, true, true, 0...4]
        assert { !gy_strided.contiguous? }
        gx = @x.pooling_backward(@max, y, gy_strided, 2)
        assert { gx == @x.pooling_backward(@max, y, dtype.new(1, 1, 4, 4).seq(1), 2) }
        assert { gx.to_a.flatten.count { |v| v != 0 } == y.size }
      end
    end

    sub_test_case "a stride of zero" do
      setup do
        @x = dtype.new(1, 1, 8, 8).seq
        @w = dtype.new(1, 1, 2, 2).seq
        @max = Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX
      end

      # ruby.h defines NDEBUG, so the assert that used to guard the division in
      # GetConvOutDim never ran and the process took a SIGFPE instead
      test "the conv entry points reject it #{dtype}" do
        assert_raise(ArgumentError) { @x.conv(@w, stride: 0) }
        assert_raise(ArgumentError) { @x.conv_transpose(@w, stride: 0) }
        assert_raise(ArgumentError) { @x.conv_grad_w(@x.conv(@w), @w.shape, stride: 0) }
      end

      test "the pooling entry points reject it #{dtype}" do
        assert_raise(ArgumentError) { @x.pooling_forward(@max, 0) }
        assert_raise(ArgumentError) { @x.pooling_forward(@max, 2, stride: 0) }
        assert_raise(ArgumentError) { @x.max_pool(0) }
        assert_raise(ArgumentError) { @x.avg_pool(0) }
        y = @x.pooling_forward(@max, 2)
        assert_raise(ArgumentError) { @x.pooling_backward(@max, y, dtype.ones(*y.shape), 2, stride: 0) }
      end

      test "a positive stride still goes through #{dtype}" do
        assert { @x.conv(@w, stride: 2).shape == [1, 1, 4, 4] }
        assert { @x.conv_transpose(@w, stride: 2).shape == [1, 1, 16, 16] }
        assert { @x.pooling_forward(@max, 2).shape == [1, 1, 4, 4] }
      end
    end

    sub_test_case "given output arrays" do
      setup do
        @x_shape = [2, 3, 5, 3]
        @reduced_shape = [1].concat(@x_shape[1..-1])
        @w_shape = [2, 3, 2, 3]
        @ksize = [3, 3]
        @x = dtype.ones(*@x_shape) * 3
        @gamma = param_type.ones(*@reduced_shape) * 2
        @beta = param_type.ones(*@reduced_shape)
        @mean = param_type.ones(*@reduced_shape)
        @var = param_type.ones(*@reduced_shape)
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
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm_backward(@gamma, @gy, ggamma: param_type.zeros(1)) }
        assert_raise(Cumo::NArray::ShapeError) { @x.batch_norm_backward(@gamma, @gy, gbeta: param_type.zeros(1)) }
        gx = dtype.zeros(*@x_shape)
        ggamma = param_type.zeros(*@reduced_shape)
        gbeta = param_type.zeros(*@reduced_shape)
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

  if float_types.include?(Cumo::HFloat)
   hf = Cumo::HFloat
   sf = Cumo::SFloat

   sub_test_case "HFloat" do

    def seq(klass, shape, modulo)
      klass.cast(Cumo::Int32.new(*shape).seq % modulo)
    end

    # cuDNN picks its algorithm by benchmarking, and half is allowed tensor
    # cores, so which one wins varies between processes and so does the last
    # place of the answer. Everything here is compared against SFloat within
    # half's own precision rather than to the bit.
    def assert_close_to_sfloat(got, ref)
      scale = [ref.abs.max.to_a.first, 1.0].max
      assert_in_delta 0.0, (Cumo::SFloat.cast(got) - ref).abs.max.to_a.first, scale * 2.0**-9
    end

    test "a convolution answers what SFloat answers" do
      x = seq(hf, [2, 8, 6, 6], 4)
      w = seq(hf, [8, 8, 1, 1], 3)
      b = seq(hf, [8], 3)
      assert_close_to_sfloat x.conv(w), sf.cast(x).conv(sf.cast(w))
      assert_close_to_sfloat x.conv(w, b: b), sf.cast(x).conv(sf.cast(w), b: sf.cast(b))
    end

    test "a 3x3 convolution and its two gradients answer what SFloat answers" do
      x = seq(hf, [2, 8, 8, 8], 4)
      w = seq(hf, [8, 8, 3, 3], 3)
      assert_close_to_sfloat x.conv(w), sf.cast(x).conv(sf.cast(w))
      gy = seq(hf, x.conv(w).shape, 3)
      assert_close_to_sfloat x.conv_grad_w(gy, w.shape), sf.cast(x).conv_grad_w(sf.cast(gy), w.shape)
      assert_close_to_sfloat gy.conv_transpose(w, out_size: [8, 8]),
                             sf.cast(gy).conv_transpose(sf.cast(w), out_size: [8, 8])
    end

    test "pooling answers what SFloat answers, to the bit" do
      x = seq(hf, [2, 3, 8, 8], 7)
      y = x.max_pool(2, stride: 2)
      assert_equal sf.cast(x).max_pool(2, stride: 2).to_a, y.to_a
      gy = seq(hf, y.shape, 3)
      assert_equal sf.cast(x).max_pool_backward(sf.cast(y), sf.cast(gy), 2, stride: 2).to_a,
                   x.max_pool_backward(y, gy, 2, stride: 2).to_a
      assert_equal sf.cast(x).avg_pool(2, stride: 2).to_a, x.avg_pool(2, stride: 2).to_a
    end

    test "the batch norm parameters are SFloat, not HFloat" do
      x = seq(hf, [2, 4, 3, 3], 5)
      g = sf.ones(4)
      b = sf.zeros(4)
      axis = [0, 2, 3]
      assert_equal hf, x.batch_norm(g, b, axis: axis).class
      assert_close_to_sfloat x.batch_norm(g, b, axis: axis), sf.cast(x).batch_norm(g, b, axis: axis)
      gx, ggamma, gbeta = x.batch_norm_backward(g, seq(hf, x.shape, 3), axis: axis)
      assert_equal [hf, sf, sf], [gx.class, ggamma.class, gbeta.class]
    end

    test "a parameter of x's own class is refused by name" do
      x = seq(hf, [2, 4, 3, 3], 5)
      e = assert_raise(TypeError) { x.batch_norm(hf.ones(4), sf.zeros(4), axis: [0, 2, 3]) }
      assert_equal "gamma must be Cumo::SFloat, not Cumo::HFloat", e.message
      e = assert_raise(TypeError) { x.batch_norm(sf.ones(4), hf.zeros(4), axis: [0, 2, 3]) }
      assert_equal "beta must be Cumo::SFloat, not Cumo::HFloat", e.message
    end

    test "the workspace ceiling is a positive byte count or the default" do
      assert_operator Cumo::CUDA::CUDNN.max_workspace_size, :>, 0
      assert_equal ENV["CUMO_CUDNN_MAX_WORKSPACE_SIZE"].to_i, Cumo::CUDA::CUDNN.max_workspace_size if
        ENV["CUMO_CUDNN_MAX_WORKSPACE_SIZE"].to_s =~ /\A[1-9][0-9]*\z/
    end

    test "the convolution accumulates over more channels than half can count" do
      # 40000 ones summed in half would stop at 2048
      x = hf.ones(1, 40_000, 1, 1)
      w = hf.ones(1, 40_000, 1, 1)
      assert_equal 40_000.0, x.conv(w).to_a.flatten.first
    end
   end
  end
end
