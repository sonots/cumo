# frozen_string_literal: true

require_relative "../test_helper"
require "tempfile"

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

    # Destroying the context leaves every pointer allocated under it invalid, so
    # cudaFree fails for the rest of the process. That has to be reported rather
    # than raised: the free runs from the GC sweep, where a raise surfaces at
    # whatever line happened to trigger the collection. Needs a fresh
    # interpreter because it makes CUDA unusable for good.
    def test_a_failing_free_from_the_gc_hook_does_not_raise
      lib = File.expand_path("../../lib", __dir__)
      script = <<~'RUBY'
        require "cumo/narray"
        require "fiddle"

        # Unbuffered, or the answer is lost with the process when the raise the
        # test is about takes the interpreter down with it.
        STDOUT.sync = true

        name = %w[cuCtxDestroy_v2 cuCtxDestroy].find do |sym|
          begin
            Fiddle::Handle::DEFAULT[sym]
            true
          rescue Fiddle::DLError
            false
          end
        end
        if name.nil?
          print "no-cuCtxDestroy"
          exit
        end
        destroy = Fiddle::Function.new(Fiddle::Handle::DEFAULT[name],
                                       [Fiddle::TYPE_VOIDP], Fiddle::TYPE_INT)

        def make_garbage
          200.times { Cumo::DFloat.new(4096).seq }
          nil
        end

        # A pooled chunk goes back to the free list without a CUDA call, so the
        # free only reaches cudaFree for memory allocated with the pool off.
        Cumo::CUDA::MemoryPool.disable

        GC.disable
        make_garbage
        ctx = Cumo::CUDA::Driver.cuCtxGetCurrent

        STDERR.reopen(ARGV.fetch(0), "w")
        destroy.call(ctx)
        GC.enable
        begin
          GC.start
          print "ok"
        rescue Exception => e
          print "raised #{e.class}"
        end
      RUBY

      Tempfile.create("cumo-free-hook") do |log|
        out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script, log.path], &:read)
        omit("cuCtxDestroy is not available") if out == "no-cuCtxDestroy"
        assert_equal("ok", out)
        # Without this the test passes even when nothing was freed at all.
        assert_match(/failed to free device memory/, log.read)
      end
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
