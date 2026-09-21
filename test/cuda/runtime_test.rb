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

    # cumo holds a context from the moment it loads so that a driver call made
    # before any array operation has one. Creating its own rather than
    # retaining the primary one used to strand everything built before the
    # first cudaSetDevice, because that binds the primary context: a cuBLAS
    # handle kept recording into an event from a context it no longer ran in,
    # and every gemm after it failed.
    #
    # It takes a fresh process: once anything in this one has set the device,
    # the context is already the primary one and the order cannot be built
    # again.
    def test_cudaSetDevice_leaves_a_cublas_handle_usable
      script = <<~RUBY
        require "cumo/narray"
        a = Cumo::SFloat[[1.0, 2.0], [3.0, 4.0]]
        a.dot(a)
        Cumo::CUDA::Runtime.cudaSetDevice(0)
        c = Cumo::SComplex[[1.0, 2.0], [3.0, 4.0]]
        c.dot(c)
        print "ok"
      RUBY
      includes = $LOAD_PATH.map { |p| ["-I", p] }.flatten
      out = IO.popen([RbConfig.ruby, *includes, "-e", script], err: [:child, :out], &:read)
      assert_equal "ok", out
    end

    def test_cudaGetDeviceCount
      assert { Runtime.cudaGetDeviceCount.is_a?(Integer) }
    end

    def test_cudaDeviceSynchronize
      assert_nothing_raised { Runtime.cudaDeviceSynchronize }
    end

    # A device never has peer access to itself, and a device that is not
    # there is reported rather than answered. A report must not be left as
    # the last error for the next kernel launch to find.
    def test_cudaDeviceCanAccessPeer
      n = Runtime.cudaGetDeviceCount
      n.times do |d|
        assert_equal(0, Runtime.cudaDeviceCanAccessPeer(d, d))
        n.times { |p| assert_include([0, 1], Runtime.cudaDeviceCanAccessPeer(d, p)) if p != d }
      end
      assert_raise(Cumo::CUDA::RuntimeError) { Runtime.cudaDeviceCanAccessPeer(0, n) }
      assert_raise(Cumo::CUDA::RuntimeError) { Runtime.cudaDeviceCanAccessPeer(n, 0) }
      assert_raise(TypeError) { Runtime.cudaDeviceCanAccessPeer(0, nil) }
      assert_equal(6.0, Cumo::SFloat.new(4).seq.sum.to_f)
    end

    def test_a_failed_call_does_not_fail_the_next_launch
      n = Runtime.cudaGetDeviceCount
      assert_raise(Cumo::CUDA::RuntimeError) { Runtime.cudaSetDevice(n) }
      assert_equal(6.0, Cumo::SFloat.new(4).seq.sum.to_f)
    end
  end
end
