$LOAD_PATH.unshift File.expand_path("../../lib", __FILE__)
require "cumo"
Cumo::CUDA::Runtime.cudaSetDevice(ENV['CUDA_DEVICE'] || 0)

require "pry"
require "test/unit"
