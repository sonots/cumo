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

    # Both handlers around pool.Malloc used to raise from inside the catch, and
    # rb_raise longjmps: __cxa_end_catch never runs, so the exception object is
    # never destroyed. An allocation which fails twice has two of them live.
    # In a fresh interpreter, because the retry runs GC.start and marking this
    # suite's heap makes each failed allocation eight times more expensive.
    def test_a_failing_allocation_does_not_leak_the_cxx_exception
      omit "needs /proc/self/status" unless File.readable?("/proc/self/status")
      lib = File.expand_path("../../lib", __dir__)
      script = <<~'RUBY'
        require "cumo/narray"

        rss = -> { Integer(File.read("/proc/self/status")[/VmRSS:\s+(\d+)/, 1], 10) }
        too_big = (2**61) - 1
        burn = lambda do |n|
          n.times do
            Cumo::DFloat.new(too_big).allocate
          rescue Cumo::CUDA::OutOfMemoryError
            nil
          end
        end

        burn.call(200)
        GC.start
        before = rss.call
        burn.call(2000)
        GC.start
        print "rss:#{rss.call - before}"
      RUBY

      out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], &:read)
      assert_match(/\Arss:-?\d+\z/, out)
      # the leak alone was 1060 KB for 2000 failed allocations
      assert { Integer(out[/-?\d+/], 10) < 256 }
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
