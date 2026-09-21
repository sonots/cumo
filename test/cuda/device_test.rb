# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class DeviceTest < Test::Unit::TestCase
    def test_initialize
      assert { Device.new(0).id == 0 }
      assert { Device.new.id.is_a?(Integer) }
    end

    def test_use
      assert_nothing_raised { Device.new(0).use }
    end

    def test_with
      Device.new(0).with do
        assert { Device.new.id == 0 }
      end
    end

    # The switch used to happen only where there was nothing to switch to, so
    # asking for another device left the block running on the one it started on.
    def test_with_switches_to_another_device
      count = Runtime.cudaGetDeviceCount
      omit("needs a second device") if count < 2

      Device.new(1).with do
        assert { Device.new.id == 1 }
      end
      assert { Device.new.id == 0 }
    end

    # Where there is one device, a second one is what shows whether the switch
    # is made at all: making it is what fails, and skipping it does not. The
    # failure stays in the slot the next CUDA call reads, so it goes in a child.
    def test_with_reports_a_device_that_is_not_there
      lib = File.expand_path("../../lib", __dir__)
      script = <<~RUBY
        require "cumo/narray"
        count = Cumo::CUDA::Runtime.cudaGetDeviceCount
        begin
          Cumo::CUDA::Device.new(count).with { print "ran:" }
          print "no-error"
        rescue Cumo::CUDA::RuntimeError
          print "raised"
        end
        print ":back-on-\#{Cumo::CUDA::Runtime.cudaGetDevice}"
      RUBY

      out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
      assert_equal("raised:back-on-0", out)
    end

    def test_synchronize
      assert_nothing_raised { Device.new.synchronize }
    end

    def test_can_access_peer
      d = Device.new(0)
      assert_equal(false, d.can_access_peer?(0))
      assert_equal(false, d.can_access_peer?(d))
      assert_raise(Cumo::CUDA::RuntimeError) { d.can_access_peer?(Runtime.cudaGetDeviceCount) }
      assert_raise(TypeError) { d.can_access_peer?(0.9) }
      assert_raise(TypeError) { d.can_access_peer?("0") }
    end

    def test_compute_capability
      capability = `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.strip
      assert { Device.new.compute_capability == capability.delete('.') }
    end
  end
end
