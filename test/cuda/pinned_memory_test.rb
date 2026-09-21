# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class PinnedMemoryTest < Test::Unit::TestCase
    test "a buffer holds bytes that are read and written within its size" do
      p = PinnedMemory.new(16)
      assert_equal(16, p.bytesize)
      assert_equal(16, p.read.bytesize)
      p.write([1.5, 2.5, 3.5, 4.5].pack("f*"))
      assert_equal([1.5, 2.5, 3.5, 4.5], p.read.unpack("f*"))
      assert_equal([3.5], p.read(4, 8).unpack("f*"))
      p.write([9.0].pack("f"), 12)
      assert_equal([1.5, 2.5, 3.5, 9.0], p.to_binary.unpack("f*"))
      assert_raise(RangeError) { p.read(4, 13) }
      assert_raise(RangeError) { p.write("x" * 17) }
      assert_raise(RangeError) { p.write("x", 16) }
      assert_raise(ArgumentError) { PinnedMemory.new(0) }
      q = PinnedMemory.from_binary("abcd")
      assert_equal("abcd", q.read)
      p.free
      assert_raise(ArgumentError) { p.read }
      assert_raise(ArgumentError) { p.write("x") }
      assert_nothing_raised { p.free }
      assert_raise(ArgumentError) { Runtime.cudaFreeHost(12345) }
    end

    test "set copies a pinned buffer into an array and get copies an array out" do
      a = Cumo::SFloat.new(1000).seq(0, 0.5)
      src = PinnedMemory.from_binary(a.to_binary)
      b = Cumo::SFloat.new(1000)
      assert_same(b, b.set(src))
      Runtime.cudaDeviceSynchronize
      assert_equal(a.to_a, b.to_a)

      dst = PinnedMemory.new(a.byte_size)
      assert_same(dst, (a * 2).get(dst))
      Runtime.cudaDeviceSynchronize
      assert_equal((a * 2).to_a, Cumo::SFloat.from_binary(dst.read, [1000]).to_a)

      assert_equal(a.to_binary, a.get)
      c = Cumo::SFloat.new(1000)
      assert_same(c, c.set(a.to_binary))
      assert_equal(a.to_a, c.to_a)
    end

    test "a copy on a stream is ordered by the stream and can be timed" do
      a = Cumo::SFloat.new(1 << 20).seq
      dst = PinnedMemory.new(a.byte_size)
      s = Stream.new(non_blocking: true)
      start = Event.new
      stop = Event.new
      s.with do
        start.record
        a.get(dst)
        stop.record
      end
      assert { stop.done? }
      assert_operator Cumo::CUDA.get_elapsed_time(start, stop), :>, 0.0
      assert_equal(a[0...5].to_a, dst.read(20).unpack("f*"))

      b = Cumo::SFloat.new(1 << 20)
      t = Stream.new(non_blocking: true)
      b.set(dst, stream: t)
      t.synchronize
      assert_equal(a.to_a, b.to_a)
    end

    test "read, write and free wait for the copy in flight, and the array is kept until then" do
      a = Cumo::SFloat.new(1 << 22).seq
      dst = PinnedMemory.new(a.byte_size)
      s = Stream.new(non_blocking: true)
      a.get(dst, stream: s)
      assert_equal(a[-4..-1].to_a, dst.read(16, a.byte_size - 16).unpack("f*"))
      assert_same(a, dst.instance_variable_get(:@busy_with)) if dst.instance_variable_get(:@state)[1]
      src = PinnedMemory.from_binary(a.to_binary)
      b = Cumo::SFloat.new(1 << 22)
      b.set(src, stream: s)
      src.write([9.0].pack("f"))
      assert_equal(0.0, b[0].to_f)
      assert_nothing_raised { src.free }
      c = Cumo::SFloat.new(1 << 22)
      c.set(PinnedMemory.from_binary(a.to_binary), stream: s)
      GC.start
      s.synchronize
      assert_equal(a[100...110].to_a, c[100...110].to_a)
    end

    test "a contiguous view with an offset, a 0-dimensional view, a frozen array and an unallocated one" do
      a = Cumo::SFloat.new(10).seq
      p4 = PinnedMemory.new(16)
      a[2..5].get(p4)
      p4.synchronize
      assert_equal([2.0, 3.0, 4.0, 5.0], p4.read.unpack("f*"))
      a[6..9].set(PinnedMemory.from_binary([1.0, 2.0, 3.0, 4.0].pack("f*")))
      Runtime.cudaDeviceSynchronize
      assert_equal([0, 1, 2, 3, 4, 5, 1, 2, 3, 4].map(&:to_f), a.to_a)
      p1 = PinnedMemory.new(4)
      a[3].get(p1)
      assert_equal([3.0], p1.read.unpack("f*"))
      a[3].set(PinnedMemory.from_binary([7.5].pack("f")))
      assert_equal(7.5, a[3].to_f)
      assert_raise(::RuntimeError) { a.freeze.set(PinnedMemory.new(40)) }
      assert_raise(::RuntimeError) { Cumo::SFloat.new(10).get(PinnedMemory.new(40)) }
      fresh = Cumo::SFloat.new(4)
      fresh.set(PinnedMemory.from_binary([1.0, 2.0, 3.0, 4.0].pack("f*")))
      assert_equal([1.0, 2.0, 3.0, 4.0], fresh.to_a)
    end

    test "what cannot be copied is refused before the copy" do
      a = Cumo::SFloat.new(10).seq
      p = PinnedMemory.new(40)
      assert_raise(ArgumentError) { Cumo::SFloat.new(11).set(p) }
      assert_raise(ArgumentError) { Cumo::SFloat.new(10, 2)[true, 0].set(p) }
      assert_raise(ArgumentError) { Cumo::SFloat.new(11).get(p) }
      assert_raise(TypeError) { Cumo::Bit.new(320).set(p) }
      assert_raise(TypeError) { Cumo::RObject.new(5).get(p) }
      assert_raise(TypeError) { a.set(3) }
      assert_raise(TypeError) { a.get(3) }
      assert_raise(ArgumentError) { a.set(a.to_binary, stream: Stream.null) }
      assert_raise(TypeError) { a.get(p, stream: 0) }
      assert_raise(TypeError) { a.set(p, stream: Runtime.current_stream) }
      assert_raise(ArgumentError) { a.get(stream: Stream.null) }
      freed = PinnedMemory.new(40)
      freed.free
      assert_raise(ArgumentError) { a.set(freed) }
      assert_raise(ArgumentError) { a.get(freed) }
      destroyed = Stream.new
      destroyed.destroy
      assert_raise(ArgumentError) { a.get(p, stream: destroyed) }
      assert_raise(ArgumentError) { Runtime.memcpy_pinned_to_narray(12345, a, nil) }
      10.times { PinnedMemory.new(1 << 16) }
      GC.start
      assert_nothing_raised { PinnedMemory.new(8).write("12345678") }
    end
  end
end
