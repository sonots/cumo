# frozen_string_literal: true

module Cumo
  [BFloat, HFloat, SFloat, DFloat].each do |klass|
    klass.class_eval do
      alias_method :conv_backward_data, :conv_transpose
      alias_method :conv_backward_filter, :conv_grad_w

      def max_pool(*args, **kwargs)
        self.pooling_forward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
      end

      def max_pool_backward(*args, **kwargs)
        self.pooling_backward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
      end

      def avg_pool(*args, **kwargs)
        pad_value = kwargs.delete(:pad_value)
        if pad_value == 0
          mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        elsif pad_value.nil?
          mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
        else
          raise "pad_value: #{pad_value} is not supported"
        end
        self.pooling_forward(mode, *args, **kwargs)
      end

      def avg_pool_backward(*args, **kwargs)
        pad_value = kwargs.delete(:pad_value)
        if pad_value == 0
          mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        elsif pad_value.nil?
          mode = Cumo::CUDA::CUDNN::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
        else
          raise "pad_value: #{pad_value} is not supported"
        end
        self.pooling_backward(mode, *args, **kwargs)
      end
    end
  end
end

module Cumo
  module CUDA
    module CUDNN
      class << self
        # Returns a.conv(*args, **kwargs).
        # @see Cumo::DFloat#conv
        def conv(a, *args, **kwargs)
          a.conv(*args, **kwargs)
        end

        # Returns a.conv_transpose(*args, **kwargs).
        # @see Cumo::DFloat#conv_transpose
        def conv_transpose(a, *args, **kwargs)
          a.conv_transpose(*args, **kwargs)
        end
        alias_method :conv_backward_data, :conv_transpose

        # Returns a.conv_grad_w(*args, **kwargs).
        # @see Cumo::DFloat#conv_grad_w
        def conv_grad_w(a, *args, **kwargs)
          a.conv_grad_w(*args, **kwargs)
        end
        alias_method :conv_backward_filter, :conv_grad_w

        # Returns a.batch_norm(*args, **kwargs).
        # @see Cumo::DFloat#batch_norm
        def batch_norm(a, *args, **kwargs)
          a.batch_norm(*args, **kwargs)
        end

        # Returns a.batch_norm_backward(*args, **kwargs).
        # @see Cumo::DFloat#batch_norm_backward
        def batch_norm_backward(a, *args, **kwargs)
          a.batch_norm_backward(*args, **kwargs)
        end

        # Max pooling, which returns
        # a.pooling_forward(CUDNN_POOLING_MAX, kernel_size, ...).
        # @see Cumo::DFloat#pooling_forward
        def max_pool(a, *args, **kwargs)
          a.max_pool(*args, **kwargs)
        end

        # The gradient of max_pool, which returns
        # a.pooling_backward(CUDNN_POOLING_MAX, y, gy, kernel_size, ...).
        # @see Cumo::DFloat#pooling_backward
        def max_pool_backward(a, *args, **kwargs)
          a.max_pool_backward(*args, **kwargs)
        end

        # Average pooling, which returns a.pooling_forward with an average mode.
        # pad_value: nil, the default, leaves the padding out of the average,
        # and pad_value: 0 counts it as zeros. Any other value raises.
        # @see Cumo::DFloat#pooling_forward
        def avg_pool(a, *args, **kwargs)
          a.avg_pool(*args, **kwargs)
        end

        # The gradient of avg_pool, with the same pad_value:.
        # @see Cumo::DFloat#pooling_backward
        def avg_pool_backward(a, *args, **kwargs)
          a.avg_pool_backward(*args, **kwargs)
        end
      end
    end
  end
end
