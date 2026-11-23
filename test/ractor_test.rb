# frozen_string_literal: true

require "timeout"
require_relative "test_helper"

class NArrayRactorTest < CumoTestBase
  if respond_to?(:ractor)
    ractor keep: true

    def test_rand_in_two_ractors
      rs = 2.times.map { Ractor.new { 20.times.map { Cumo::DFloat.new(1000).rand.to_a } } }
      a, b = Timeout.timeout(60) { rs.map { |r| r.respond_to?(:take) ? r.take : r.value } }
      assert_empty(a & b)
    end

    # searchsorted picks its type from tables, which a Ractor other than the
    # main one can read only once they are frozen through.
    def test_searchsorted_in_a_ractor
      r = Ractor.new do
        promoted = Cumo::UInt32[1, 3_000_000_000].searchsorted(Cumo::Int32[5]).to_a
        sorted = Cumo::DFloat[3, 1, 2].searchsorted([2.5], sorter: [1, 2, 0]).to_a
        [promoted, sorted]
      end
      assert_equal([[1], [2]], Timeout.timeout(60) { r.respond_to?(:take) ? r.take : r.value })
    end

    # A destroy reaches the live set, but not the threads of another Ractor
    # that have the handle current, and a launch on the released stream took
    # the process down with it.
    def test_launch_on_a_stream_another_ractor_destroyed
      stream = Cumo::CUDA::Stream.new(non_blocking: true)
      handle = stream.handle
      stream.use
      r = Ractor.new(handle) do |h|
        Cumo::CUDA::Runtime.cudaStreamDestroy(h)
        :destroyed
      end
      assert_equal(:destroyed, Timeout.timeout(60) { r.respond_to?(:take) ? r.take : r.value })
      assert_raise(ArgumentError) { Cumo::SFloat.new(16).seq }
      # and the thread goes on once it stops naming the stream
      Cumo::CUDA::Runtime.current_stream = 0
      assert_equal(120.0, Cumo::SFloat.new(16).seq.sum.to_f)
    end

    data(:dtype, TYPES, keep: true)
    def test_non_frozen(data)
      dtype = data.fetch(:dtype)
      ary = random_array(dtype)
      r = Ractor.new(ary) { |x| x }
      ary2 = r.respond_to?(:take) ? r.take : r.value
      assert_equal(ary, ary2)
      assert_not_same(ary, ary2)
    end

    def test_frozen(data)
      dtype = data.fetch(:dtype)
      ary1 = random_array(dtype)
      ary1.freeze
      r = Ractor.new(ary1) do |ary2|
        [ary2, ary2 * 10]
      end
      ary2, res = r.respond_to?(:take) ? r.take : r.value
      assert_equal((dtype != Cumo::RObject),
                   ary1.equal?(ary2))
      assert_equal(ary1 * 10, res)
    end

    def test_parallel(data)
      dtype = data.fetch(:dtype)
      ary1 = random_array(dtype, 100000)
      r1 = Ractor.new(ary1) do |ary2|
        ary2 * 10
      end
      r2 = Ractor.new(ary1) do |ary4|
        ary4 * 10
      end
      result1 = r1.respond_to?(:take) ? r1.take : r1.value
      result2 = r2.respond_to?(:take) ? r2.take : r2.value
      assert_equal(result1, result2)
    end

    def random_array(dtype, n=1000)
      case dtype
      when Cumo::DFloat, Cumo::SFloat, Cumo::DComplex, Cumo::SComplex
        dtype.new(n).rand_norm
      else
        dtype.new(n).rand(10)
      end
    end
  end
end
