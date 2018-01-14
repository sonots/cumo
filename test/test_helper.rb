$LOAD_PATH.unshift File.expand_path("../../lib", __FILE__)
require "cumo"
Cumo::CUDA::Runtime.cudaSetDevice(Integer(ENV['CUDA_DEVICE'] || 0))

require "pry"
require "test/unit"
