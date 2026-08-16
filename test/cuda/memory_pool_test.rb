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

    # Device memory does not pass through ruby_xmalloc, so unless the GC is told
    # about it these tiny Ruby objects never trigger one and the pool grows to
    # the total churn instead of the live set.
    def test_dropped_arrays_do_not_grow_the_pool_by_their_total_size
      MemoryPool.enable
      GC.start
      MemoryPool.free_all_blocks
      base = MemoryPool.total_bytes
      # 1 MiB each, 1 GiB of churn against a live set of one array
      1024.times { Cumo::SFloat.new(1 << 18).allocate }
      GC.start
      assert { MemoryPool.total_bytes - base < 256 << 20 }
    end

    def test_dropped_bit_arrays_do_not_grow_the_pool_by_their_total_size
      MemoryPool.enable
      GC.start
      MemoryPool.free_all_blocks
      base = MemoryPool.total_bytes
      1024.times { Cumo::Bit.new(1 << 23).allocate }
      GC.start
      assert { MemoryPool.total_bytes - base < 256 << 20 }
    end

    def test_malloc_with_size_whose_rounding_overflows
      MemoryPool.enable
      # 8 * (2**61 - 1) is SIZE_MAX - 7. Rounding it up to a multiple of the
      # 512 byte alignment wrapped around to 0, so the pool handed out a chunk
      # of another allocation instead of reporting the failure.
      assert_raise(OutOfMemoryError) { Cumo::DFloat.new(2**61 - 1).allocate }
      # the pool must be left usable
      assert { Cumo::DFloat.new(1024).seq.sum == 1024 * 1023 / 2 }
    end
  end
end
