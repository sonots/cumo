# frozen_string_literal: true

ENV['CUDA_MODULE_LOADING'] ||= 'LAZY'

require_relative 'cumo.so'
require_relative 'cumo/cuda'
require_relative 'cumo/narray/extra'

[Cumo::SFloat, Cumo::DFloat, Cumo::SComplex, Cumo::DComplex].each do |klass|
  klass.__send__(:private, :abs_sum, :abs_max, :abs_min)
end
