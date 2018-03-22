require_relative "../test_helper"

module Cumo::CUDA
  class MemoryPoolTest < Test::Unit::TestCase
    class << self
      def startup
        @orig_state = MemoryPool.enabled?
      end

      def shutdown
        @orig_state ? MemoryPool.enable : MemoryPool.disable
      end
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
      assert { MemoryPool.n_free_blocks == 0 }
    end

    def test_used_bytes
      assert { MemoryPool.used_bytes == 0 }
    end

    def test_free_bytes
      assert { MemoryPool.free_bytes == 0 }
    end

    def test_total_bytes
      assert { MemoryPool.total_bytes == 0 }
    end
  end
end
