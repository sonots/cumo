# frozen_string_literal: true

ENV['CUDA_MODULE_LOADING'] ||= 'LAZY'

require_relative 'cumo.so'
require_relative 'cumo/cuda'
require_relative 'cumo/narray/extra'
