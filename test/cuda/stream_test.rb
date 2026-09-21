# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class StreamTest < Test::Unit::TestCase
    # Every launch, copy and library call goes to the current stream. Under a
    # non-blocking stream nothing orders itself after stream 0 any more, so
    # the results only agree if every one of them was moved.
    def on_stream(stream)
      before = Runtime.current_stream
      Runtime.current_stream = stream
      yield
    ensure
      Runtime.current_stream = before
    end

    def battery
      a = Cumo::SFloat.new(300, 40).seq(0, 0.5)
      b = Cumo::SFloat.new(40, 300).seq(1, -0.25)
      idx = Cumo::Int32[5, 1, 299, 7]
      bits = a.gt(50)
      {
        elementwise: (a * 2 + 1).to_a,
        reduce: a.sum(axis: 0).to_a,
        gemm: a.gemm(b)[0...3, 0...3].to_a,
        sort: a.sort(axis: 1)[0, 0...5].to_a,
        cumsum: a[0, true].cumsum.to_a,
        where: bits.where[0...5].to_a,
        count: bits.count_true.to_i,
        index: a[idx, true][true, 0].to_a,
        bit: bits[3, 7],
        view_end: a[(0...300).step(7), true][-1, -1],
        mulsum: a.mulsum(a, axis: 1)[0...3].to_a,
      }
    end

    test "every kernel, copy and library call answers the same on a stream of its own" do
      want = battery
      s = Runtime.cudaStreamCreateWithFlags(Runtime::CUDA_STREAM_NON_BLOCKING)
      begin
        got = on_stream(s) do
          assert_equal(s, Runtime.current_stream)
          battery
        end
        Runtime.cudaStreamSynchronize(s)
      ensure
        Runtime.cudaStreamDestroy(s)
      end
      assert_equal(0, Runtime.current_stream)
      assert_equal(want, got)
    end

    test "a stream is created, waited for and destroyed, and only a live one is taken" do
      s = Runtime.cudaStreamCreateWithFlags(Runtime::CUDA_STREAM_DEFAULT)
      assert_nothing_raised { Runtime.cudaStreamSynchronize(s) }
      assert_nothing_raised { Runtime.cudaStreamSynchronize(0) }
      assert_equal(true, Runtime.cudaStreamQuery(s))
      Runtime.current_stream = s
      assert_raise(ArgumentError) { Runtime.cudaStreamDestroy(s) }
      Runtime.current_stream = 0
      Runtime.cudaStreamDestroy(s)
      assert_raise(ArgumentError) { Runtime.cudaStreamDestroy(s) }
      assert_raise(ArgumentError) { Runtime.cudaStreamSynchronize(s) }
      assert_raise(ArgumentError) { Runtime.current_stream = s }
      assert_raise(ArgumentError) { Runtime.current_stream = 12345 }
      assert_equal(0, Runtime.current_stream)
    end

    test "the current stream is per thread" do
      s = Runtime.cudaStreamCreateWithFlags(Runtime::CUDA_STREAM_NON_BLOCKING)
      begin
        Runtime.current_stream = s
        assert_equal(0, Thread.new { Runtime.current_stream }.value)
      ensure
        Runtime.current_stream = 0
        Runtime.cudaStreamDestroy(s)
      end
    end

    test "Stream#with runs the block on the stream and puts the previous one back" do
      s = Stream.new(non_blocking: true)
      assert { Stream.current.null? }
      inner = nil
      s.with do |given|
        assert_same(s, given)
        assert_same(s, Stream.current)
        assert_equal(s.ptr, Runtime.current_stream)
        inner = Stream.new
        inner.with { assert_same(inner, Stream.current) }
        assert_same(s, Stream.current)
      end
      assert { Stream.current.null? }
      assert_raise(::RuntimeError) { s.with { raise "inside" } }
      assert { Stream.current.null? }
      s.use
      assert_same(s, Stream.current)
      Stream.null.use
      assert { Stream.current.null? }
      assert_equal(0, Stream.null.ptr)
    end

    # The block's work is waited for on the way out, so what it produced can
    # be read on the null stream without a wait of the caller's.
    test "what a Stream#with block computed is complete when the block returns" do
      s = Stream.new(non_blocking: true)
      a = Cumo::SFloat.new(2048, 2048).fill(1)
      c = s.with { a.gemm(a) }
      assert { s.done? }
      assert_equal(2048.0, c[0, 0].to_f)
      assert_equal([2048.0] * 4, c[5, 0...4].to_a)
    end

    # A host read under a stream of the caller's waits for the whole device,
    # since the array may have been written on another stream. to_a waits for
    # the device anyway; a Bit element and an index array are read through
    # the stream. The kernel spins before it writes, so a read that did not
    # wait would see the zeros. On an RTX 5070 Ti a stream-only wait still
    # answered right, the driver ordering the managed copy itself, so this
    # holds the values rather than proving the wait.
    SLOW_FILL = ElementwiseKernel.new("T x, int32 spins", "T y",
                                      "T acc = x; for (int k = 0; k < spins; k++) { acc = acc * 1.0000001f + 1e-7f; } y = acc > 0 ? 42 : 43",
                                      "slow_fill")

    test "a host read inside a Stream#with waits for work queued on the null stream" do
      s = Stream.new(non_blocking: true)
      x = Cumo::SFloat.new(4096).fill(1)
      y = Cumo::SFloat.zeros(4096)
      SLOW_FILL.call(x, 20_000_000, y)
      bits = y.gt(0)
      idx = Cumo::Int32.new(3).fill(0) + 7
      s.with do
        assert_equal(1, bits.aref_cpu(4095))
        assert_equal([42.0] * 3, y[idx].to_a)
        assert_equal(42.0, y.aref_cpu(0))
      end
    end

    test "an event is recorded, waited for, timed, and orders one stream after another" do
      start = Event.new
      assert_same(start, start.record)
      a = Cumo::SFloat.new(2048, 2048).fill(1)
      c = a.gemm(a)
      stop = Event.new.record
      stop.synchronize
      assert { stop.done? }
      ms = Cumo::CUDA.get_elapsed_time(start, stop)
      assert_operator ms, :>, 0.0
      assert_operator ms, :<, 10_000.0
      assert_equal(2048.0, c[0, 0].to_f)

      s1 = Stream.new(non_blocking: true)
      s2 = Stream.new(non_blocking: true)
      x = Cumo::SFloat.new(2048, 2048).fill(1)
      y = nil
      s1.use
      y = x.gemm(x)
      done = s1.record
      s2.use
      s2.wait_event(done)
      z = y + 1
      Stream.null.use
      s2.synchronize
      assert_equal(2049.0, z[0, 0].to_f)
      assert_equal([2049.0] * 2, z[1, 0...2].to_a)

      light = Event.new(timing: false).record
      assert_nothing_raised { light.synchronize }
      assert_raise(Cumo::CUDA::RuntimeError) { Cumo::CUDA.get_elapsed_time(start, light) }
      assert_raise(TypeError) { Event.new.record(0) }
    end

    test "a Function launches on a Stream it is given" do
      mod = Module.new
      mod.load(Compiler.new.compile_using_nvrtc(<<~CUDA))
        extern "C" __global__ void twice(float* y, const float* x, int n) {
          for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) y[i] = x[i] * 2;
        }
      CUDA
      s = Stream.new(non_blocking: true)
      x = Cumo::SFloat.new(1000).seq
      y = Cumo::SFloat.zeros(1000)
      f = mod.get_function("twice")
      f.launch([y, x, [1000].pack("l")], grid: 4, block: 256, stream: s)
      s.synchronize
      assert_equal((x * 2).to_a, y.to_a)
      assert_raise(TypeError) { f.launch([y, x, [1000].pack("l")], grid: 4, block: 256, stream: s.ptr) }
      s.with do
        y.fill(0)
        f.launch([y, x, [1000].pack("l")], grid: 4, block: 256, stream: Stream.null)
        Stream.null.synchronize
      end
      assert_equal((x * 2).to_a, y.to_a)
      destroyed = Stream.new
      destroyed.destroy
      assert_raise(ArgumentError) { f.launch([y, x, [1000].pack("l")], grid: 4, block: 256, stream: destroyed) }
      mod.unload
    end

    # The block's kernels are ordered after what the previous stream had
    # queued, so an input still being written on the null stream is read
    # complete. The kernel spins before it writes, so a block that did not
    # wait would read the zeros.
    test "Stream#with orders the block after what the null stream had queued" do
      x = Cumo::SFloat.new(4096).fill(1)
      y = Cumo::SFloat.zeros(4096)
      SLOW_FILL.call(x, 20_000_000, y)
      s = Stream.new(non_blocking: true)
      z = s.with { y + 1 }
      assert_equal([43.0] * 3, z[0...3].to_a)
    end

    test "a stream that is current in another thread cannot be destroyed there or here" do
      s = Stream.new(non_blocking: true)
      s.use
      assert_raise(ArgumentError) { Thread.new { Thread.current.report_on_exception = false; s.destroy }.value }
      assert_raise(ArgumentError) { s.destroy }
      assert_equal(6.0, Cumo::SFloat.new(4).seq.sum.to_f)
      Stream.null.use
      assert_nothing_raised { s.destroy }
      assert_nil(s.ptr)
      t = Stream.new
      Thread.new { t.use; Thread.stop }.tap { |th| sleep 0.05 until th.status == "sleep" }
      assert_raise(ArgumentError) { t.destroy }
    end

    test "a destroyed stream or event refuses to be used, and a dropped one is reclaimed" do
      s = Stream.new
      e = Event.new
      s.destroy
      e.destroy
      assert_raise(ArgumentError) { s.use }
      assert_raise(ArgumentError) { s.synchronize }
      assert_raise(ArgumentError) { e.record }
      assert_raise(ArgumentError) { Event.new.record(s) }
      assert_raise(ArgumentError) { Stream.new.wait_event(e) }
      assert_raise(ArgumentError) { Cumo::CUDA.get_elapsed_time(e, Event.new) }
      assert_nothing_raised { s.destroy }
      assert_nothing_raised { Stream.null.destroy }
      assert_raise(ArgumentError) { Runtime.cudaStreamDestroy(0) }
      10.times { Stream.new(non_blocking: true); Event.new }
      GC.start
      assert_nothing_raised { Stream.new.with { Cumo::SFloat.new(4).seq.sum } }
    end
  end
end
