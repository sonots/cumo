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
  end
end
