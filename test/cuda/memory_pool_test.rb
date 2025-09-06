# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class MemoryPoolTest < Test::Unit::TestCase
    def setup
      @orig_state = MemoryPool.enabled?
    end

    def teardown
      @orig_state ? MemoryPool.enable : MemoryPool.disable
    end

    def test_enable
      MemoryPool.enable
      assert { MemoryPool.enabled? }
    end

    def test_disable
      MemoryPool.disable
      assert { !MemoryPool.enabled? }
    end

    def test_free_all_blocks
      assert_nothing_raised { MemoryPool.free_all_blocks }
    end

    def test_n_free_blocks
      assert_nothing_raised { MemoryPool.n_free_blocks }
    end

    def test_used_bytes
      assert_nothing_raised { MemoryPool.used_bytes }
    end

    def test_free_bytes
      assert_nothing_raised { MemoryPool.free_bytes }
    end

    def test_total_bytes
      assert_nothing_raised { MemoryPool.total_bytes }
    end
  end
end
