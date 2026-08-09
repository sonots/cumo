# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class DriverTest < Test::Unit::TestCase
    SOURCE = <<~CUDA
      extern "C" {
        __device__ int cumo_test_global[4] = {11, 22, 33, 44};
      }
      __global__ void cumo_test_kernel() {}
    CUDA

    def test_cuModuleGetGlobal
      ptx = Compiler.new.compile_using_nvrtc(SOURCE)
      Module.new do |mod|
        mod.load(ptx)
        # the global lives in device memory, so reading it needs a copy back
        var = mod.get_global_var("cumo_test_global")
        assert { var.bytesize == 16 }
        assert { var.unpack("l<4") == [11, 22, 33, 44] }
      end
    end
  end
end
