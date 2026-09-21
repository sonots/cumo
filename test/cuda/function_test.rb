# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class FunctionTest < Test::Unit::TestCase
    SOURCE = <<~CUDA
      extern "C" {
      __global__ void axpy(float* y, const float* x, float a, int n) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
          y[i] = a * x[i] + y[i];
        }
      }
      __global__ void wide(double* out, long long i, double d) {
        out[0] = (double)i;
        out[1] = d;
      }
      struct pair { int lo; float hi; };
      __global__ void by_value(float* out, struct pair p) {
        out[0] = (float)p.lo;
        out[1] = p.hi;
      }
      __global__ void block_sum(float* out, const float* x, int n) {
        extern __shared__ float buf[];
        int t = threadIdx.x;
        buf[t] = t < n ? x[t] : 0.0f;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
          if (t < s) buf[t] += buf[t + s];
          __syncthreads();
        }
        if (t == 0) out[0] = buf[0];
      }
      __global__ void where_am_i(int* out) {
        out[0] = gridDim.x; out[1] = gridDim.y; out[2] = gridDim.z;
        out[3] = blockDim.x; out[4] = blockDim.y; out[5] = blockDim.z;
      }
      }
    CUDA

    class << self
      def startup
        @@mod = Module.new
        @@mod.load(Compiler.new.compile_using_nvrtc(SOURCE))
      end

      def shutdown
        @@mod.unload
      end
    end

    def fn(name)
      @@mod.get_function(name)
    end

    test "a kernel reads and writes the NArrays it is handed" do
      n = 1000
      x = Cumo::SFloat.new(n).seq
      y = x * 3.0
      want = x * 2.5 + y
      fn("axpy").launch([y, x, [2.5].pack("f"), [n].pack("l")], grid: 4, block: 256)
      assert { y == want }
    end

    test "an Integer goes as a long long and a Float as a double" do
      out = Cumo::DFloat.zeros(2)
      fn("wide").launch([out, 2**40 + 3, 0.25], grid: 1, block: 1)
      assert_equal([(2**40 + 3).to_f, 0.25], out.to_a)
    end

    test "packed bytes go as a struct by value" do
      out = Cumo::SFloat.zeros(2)
      fn("by_value").launch([out, [7, 1.5].pack("lf")], grid: 1, block: 1)
      assert_equal([7.0, 1.5], out.to_a)
    end

    test "shared_mem is the dynamic shared memory the kernel gets" do
      x = Cumo::SFloat.new(200).seq
      out = Cumo::SFloat.zeros(1)
      fn("block_sum").launch([out, x, [200].pack("l")], grid: 1, block: 256, shared_mem: 256 * 4)
      assert_equal([x.sum.to_f], out.to_a)
    end

    test "grid and block take one to three sizes" do
      out = Cumo::Int32.zeros(6)
      fn("where_am_i").launch([out], grid: [2, 3], block: [4, 5, 6])
      assert_equal([2, 3, 1, 4, 5, 6], out.to_a)
      fn("where_am_i").launch([out], grid: 7, block: 8)
      assert_equal([7, 1, 1, 8, 1, 1], out.to_a)
      [[1, 1, 1, 1], [], 0, -1, 1.9, nil, (1..2), [1, 1.0]].each do |bad|
        assert_raise(ArgumentError, bad.inspect) { fn("where_am_i").launch([out], grid: bad, block: 1) }
        assert_raise(ArgumentError, bad.inspect) { fn("where_am_i").launch([out], grid: 1, block: bad) }
      end
      assert_raise(ArgumentError) { fn("where_am_i").launch([out], grid: 1, block: 1, shared_mem: -1) }
      assert_raise(ArgumentError) { fn("where_am_i").launch([out], grid: 1, block: 1, shared_mem: 1.5) }
    end

    test "an NArray that is not allocated yet is allocated on the way" do
      out = Cumo::DFloat.new(2)
      fn("wide").launch([out, 1, 2.0], grid: 1, block: 1)
      assert_equal([1.0, 2.0], out.to_a)
    end

    # The count and the sizes come from the driver, which reports them from
    # CUDA 12.4 on.
    test "the arguments are held to the kernel's parameters" do
      omit("the driver does not report kernel parameters") if Runtime.cudaDriverGetVersion < 12040
      out = Cumo::SFloat.zeros(2)
      x = Cumo::SFloat.new(10).seq
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, "", [10].pack("l")], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, [1.5, 9.0].pack("ff"), [10].pack("l")], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, 1.5, [10].pack("l")], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, [1.5].pack("f"), 10], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, [1.5].pack("f")], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([out, x, [1.5].pack("f"), [10].pack("l"), 0], grid: 1, block: 1) }
      e = assert_raise(ArgumentError) { fn("axpy").launch([out, x, [1.5].pack("f"), 10], grid: 1, block: 1) }
      assert_equal("argument 3 is 8 bytes where the kernel takes 4", e.message)
    end

    test "an argument a kernel cannot take is refused before the launch" do
      out = Cumo::SFloat.zeros(2)
      x = Cumo::SFloat.new(10).seq
      assert_raise(TypeError) { fn("wide").launch([out, nil, 0.0], grid: 1, block: 1) }
      assert_raise(TypeError) { fn("wide").launch([out, :sym, 0.0], grid: 1, block: 1) }
      assert_raise(TypeError) { fn("axpy").launch([Cumo::Bit.new(8), x, "", ""], grid: 1, block: 1) }
      assert_raise(ArgumentError) { fn("axpy").launch([x[(0...10).step(2)], x, "", ""], grid: 1, block: 1) }
      assert_raise(::RuntimeError) { fn("axpy").launch([x.freeze, x, "", ""], grid: 1, block: 1) }
      assert_raise(TypeError) { fn("wide").launch(out, grid: 1, block: 1) }
    end

    test "a name the module does not have raises" do
      assert_raise(DriverError) { fn("no_such_kernel") }
    end

    # The driver hands a module's address out again to a later module, so a
    # function of the unloaded one has to be forgotten rather than checked
    # against its module.
    test "a function of an unloaded module cannot be launched" do
      ptx = Compiler.new.compile_using_nvrtc(SOURCE)
      mod = Module.new
      mod.load(ptx)
      f = mod.get_function("where_am_i")
      mod.unload
      out = Cumo::Int32.zeros(6)
      assert_raise(ArgumentError) { f.launch([out], grid: 1, block: 1) }
      later = 5.times.map { m = Module.new; m.load(ptx); m }
      assert_raise(ArgumentError) { f.launch([out], grid: 1, block: 1) }
      later.first.get_function("where_am_i").launch([out], grid: 3, block: 1)
      assert_equal(3, out[0])
      later.each(&:unload)
      assert_raise(ArgumentError) { Driver.cuLaunchKernel(12345, 1, 1, 1, 1, 1, 1, 0, 0, []) }
      assert_raise(ArgumentError) { Module.new.get_function("where_am_i") }
    end

    test "a stream other than 0 is refused until Cumo has streams" do
      out = Cumo::Int32.zeros(6)
      f = fn("where_am_i")
      h = f.instance_variable_get(:@ptr)
      assert_raise(ArgumentError) { Driver.cuLaunchKernel(h, 1, 1, 1, 1, 1, 1, 0, 1, [out]) }
    end
  end
end
