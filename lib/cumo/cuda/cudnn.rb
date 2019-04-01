require 'cumo'

module Cumo
  [SFloat, DFloat].each do |klass|
    klass.define_method(:max_pool) do |*args, **kwargs|
      self.pooling_forward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
    end

    klass.define_method(:average_pool) do |*args, **kwargs|
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

    klass.define_method(:max_pool_backward) do |*args, **kwargs|
      self.pooling_backward(Cumo::CUDA::CUDNN::CUDNN_POOLING_MAX, *args, **kwargs)
    end

    klass.define_method(:average_pool_backward) do |*args, **kwargs|
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
