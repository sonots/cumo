# frozen_string_literal: true

module Cumo
  module CUDA
    module CUDNN
      # The pooling methods Cumo writes in Ruby rather than in C, which pick a
      # CUDNN_POOLING_* mode and hand the rest over to pooling_forward or
      # pooling_backward. Only a build with cuDNN defines those modes, which
      # Cumo::CUDA::CUDNN.available? reports.
      module NArrayMethods
        # @overload max_pool(kernel_size, stride: kernel_size, pad: 0, y: nil)
        #   Max pooling, which returns
        #   pooling_forward(CUDNN_POOLING_MAX, kernel_size, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#pooling_forward
        def max_pool(*args, **kwargs)
          pooling_forward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
        end

        # @overload max_pool_backward(y, gy, kernel_size, stride: kernel_size, pad: 0, gx: nil)
        #   The gradient of max_pool, which returns
        #   pooling_backward(CUDNN_POOLING_MAX, y, gy, kernel_size, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#pooling_backward
        def max_pool_backward(*args, **kwargs)
          pooling_backward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
        end

        # @overload avg_pool(kernel_size, stride: kernel_size, pad: 0, pad_value: nil, y: nil)
        #   Average pooling, which returns pooling_forward with an average
        #   mode. pad_value: nil, the default, leaves the padding out of the
        #   average, and pad_value: 0 counts it as zeros.
        #   @return [Cumo::NArray]
        #   @raise [::RuntimeError] when pad_value is neither nil nor 0.
        # @see Cumo::DFloat#pooling_forward
        def avg_pool(*args, **kwargs)
          pad_value = kwargs.delete(:pad_value)
          if pad_value == 0
            mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
          elsif pad_value.nil?
            mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
          else
            raise "pad_value: #{pad_value} is not supported"
          end
          pooling_forward(mode, *args, **kwargs)
        end

        # @overload avg_pool_backward(y, gy, kernel_size, stride: kernel_size, pad: 0, pad_value: nil, gx: nil)
        #   The gradient of avg_pool, with the same pad_value:.
        #   @return [Cumo::NArray]
        #   @raise [::RuntimeError] when pad_value is neither nil nor 0.
        # @see Cumo::DFloat#pooling_backward
        def avg_pool_backward(*args, **kwargs)
          pad_value = kwargs.delete(:pad_value)
          if pad_value == 0
            mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
          elsif pad_value.nil?
            mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
          else
            raise "pad_value: #{pad_value} is not supported"
          end
          pooling_backward(mode, *args, **kwargs)
        end
      end
    end
  end

  # The four classes cuDNN has kernels for, written out one by one so that a
  # doc tool reads the same list Ruby does.
  class BFloat; include CUDA::CUDNN::NArrayMethods; end
  class HFloat; include CUDA::CUDNN::NArrayMethods; end
  class SFloat; include CUDA::CUDNN::NArrayMethods; end
  class DFloat; include CUDA::CUDNN::NArrayMethods; end
end

module Cumo
  module CUDA
    module CUDNN
      class << self
        # @overload conv(a, w, b: nil, stride: 1, pad: 0, y: nil)
        #   Returns a.conv(w, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#conv
        def conv(a, *args, **kwargs)
          a.conv(*args, **kwargs)
        end

        # @overload conv_transpose(a, w, b: nil, stride: 1, pad: 0, out_size: nil, y: nil)
        #   Returns a.conv_transpose(w, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#conv_transpose
        def conv_transpose(a, *args, **kwargs)
          a.conv_transpose(*args, **kwargs)
        end
        alias_method :conv_backward_data, :conv_transpose

        # @overload conv_grad_w(a, gy, w_shape, stride: 1, pad: 0, gw: nil)
        #   Returns a.conv_grad_w(gy, w_shape, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#conv_grad_w
        def conv_grad_w(a, *args, **kwargs)
          a.conv_grad_w(*args, **kwargs)
        end
        alias_method :conv_backward_filter, :conv_grad_w

        # @overload batch_norm(a, gamma, beta, running_mean: nil, running_var: nil, mean: nil, inv_std: nil, eps: 2e-5, decay: 0.9, axis: [0], y: nil)
        #   Returns a.batch_norm(gamma, beta, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#batch_norm
        def batch_norm(a, *args, **kwargs)
          a.batch_norm(*args, **kwargs)
        end

        # @overload batch_norm_backward(a, gamma, gy, mean: nil, inv_std: nil, eps: 2e-5, axis: [0], gx: nil, ggamma: nil, gbeta: nil)
        #   Returns a.batch_norm_backward(gamma, gy, ...).
        #   @return [Array]
        # @see Cumo::DFloat#batch_norm_backward
        def batch_norm_backward(a, *args, **kwargs)
          a.batch_norm_backward(*args, **kwargs)
        end

        # @overload max_pool(a, kernel_size, stride: kernel_size, pad: 0, y: nil)
        #   Max pooling, which returns
        #   a.pooling_forward(CUDNN_POOLING_MAX, kernel_size, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#pooling_forward
        def max_pool(a, *args, **kwargs)
          a.max_pool(*args, **kwargs)
        end

        # @overload max_pool_backward(a, y, gy, kernel_size, stride: kernel_size, pad: 0, gx: nil)
        #   The gradient of max_pool, which returns
        #   a.pooling_backward(CUDNN_POOLING_MAX, y, gy, kernel_size, ...).
        #   @return [Cumo::NArray]
        # @see Cumo::DFloat#pooling_backward
        def max_pool_backward(a, *args, **kwargs)
          a.max_pool_backward(*args, **kwargs)
        end

        # @overload avg_pool(a, kernel_size, stride: kernel_size, pad: 0, pad_value: nil, y: nil)
        #   Average pooling, which returns a.pooling_forward with an average mode.
        #   pad_value: nil, the default, leaves the padding out of the average,
        #   and pad_value: 0 counts it as zeros.
        #   @return [Cumo::NArray]
        #   @raise [::RuntimeError] when pad_value is neither nil nor 0.
        # @see Cumo::DFloat#pooling_forward
        def avg_pool(a, *args, **kwargs)
          a.avg_pool(*args, **kwargs)
        end

        # @overload avg_pool_backward(a, y, gy, kernel_size, stride: kernel_size, pad: 0, pad_value: nil, gx: nil)
        #   The gradient of avg_pool, with the same pad_value:.
        #   @return [Cumo::NArray]
        #   @raise [::RuntimeError] when pad_value is neither nil nor 0.
        # @see Cumo::DFloat#pooling_backward
        def avg_pool_backward(a, *args, **kwargs)
          a.avg_pool_backward(*args, **kwargs)
        end
      end
    end
  end
end
