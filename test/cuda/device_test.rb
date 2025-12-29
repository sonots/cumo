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

    def test_synchronize
      assert_nothing_raised { Device.new.synchronize }
    end

    def test_compute_capability
      capability = `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`.strip
      assert { Device.new.compute_capability == capability.delete('.') }
    end
  end
end
