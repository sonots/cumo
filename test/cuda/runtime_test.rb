# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class RuntimeTest < Test::Unit::TestCase
    def test_cudaDriverGetVersion
      assert { Runtime.cudaDriverGetVersion.is_a?(Integer) }
    end

    def test_cudaRuntimeGetVersion
      assert { Runtime.cudaRuntimeGetVersion.is_a?(Integer) }
    end

    def test_cudaSetDevice_cudaGetDevice
      assert_nothing_raised { Runtime.cudaSetDevice(0) }
      assert { Runtime.cudaGetDevice == 0 }
    end

    def test_cudaGetDeviceCount
      assert { Runtime.cudaGetDeviceCount.is_a?(Integer) }
    end

    def test_cudaDeviceSynchronize
      assert_nothing_raised { Runtime.cudaDeviceSynchronize }
    end
  end
end
