# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class NameExpressionTest < Test::Unit::TestCase
    CACHE_DIR = File.join(__dir__, ".kernel_cache_names")

    class << self
      def startup
        FileUtils.rm_rf(CACHE_DIR)
      end

      def shutdown
        FileUtils.rm_rf(CACHE_DIR)
        FileUtils.rm_rf("#{CACHE_DIR}_round_trip")
      end
    end

    MATRIX = <<~CUDA
      template<typename T>
      struct Matrix {
        T value[4][4];
        __device__ T& operator() (int i, int j) { return this->value[i][j]; }
        __device__ const T& operator() (int i, int j) const { return this->value[i][j]; }
      };
      template<typename T>
      __device__ Matrix<T> operator+ (const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> res;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) res(i, j) = lhs(i, j) + rhs(i, j);
        return res;
      }
      template<typename T>
      __device__ Matrix<T> operator* (const Matrix<T>& lhs, const Matrix<T>& rhs) {
        Matrix<T> res;
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
          res(i, j) = T(0);
          for (int k = 0; k < 4; k++) res(i, j) += lhs(i, k) * rhs(k, j);
        }
        return res;
      }
      template<typename T>
      __global__ void kernel(const Matrix<T>* A, const Matrix<T>* B, const Matrix<T> C, Matrix<T>* out) {
        int i = threadIdx.x;
        out[i] = A[i] * B[i] + C;
      }
    CUDA

    test "a template kernel is named by its expression and its mangled name is reported" do
      ptx, lowered = Compiler.new.compile_using_nvrtc(MATRIX, name_expressions: ["kernel<float>", "kernel<double>"])
      assert_equal(["kernel<float>", "kernel<double>"], lowered.keys)
      assert { lowered["kernel<float>"].start_with?("_Z") }
      assert_not_equal(lowered["kernel<float>"], lowered["kernel<double>"])
      assert { ptx.include?(lowered["kernel<float>"]) }
      assert_kind_of(String, Compiler.new.compile_using_nvrtc(MATRIX))
      # a name expression instantiates the template it names, so kernel<int>
      # compiles; one that names nothing does not, and one that was not
      # given cannot be looked up
      assert_nothing_raised { NVRTCProgram.new(MATRIX, name_expressions: ["kernel<int>"]).tap(&:compile).destroy }
      assert_raise(TypeError) { NVRTCProgram.new(MATRIX, name_expressions: [123]) }
      assert_kind_of(String, Compiler.new.compile_using_nvrtc(MATRIX, name_expressions: []))
      assert_raise(CompileError) { NVRTCProgram.new(MATRIX, name_expressions: ["nothing<float>"]).compile }
      prog = NVRTCProgram.new(MATRIX)
      prog.compile
      assert_raise(NVRTCError) { prog.lowered_name("kernel<float>") }
      prog.destroy
    end

    # CuPy's custom_struct/packed_matrix: arrays of a struct are arrays whose
    # trailing axes are its fields, and a struct by value is packed bytes.
    test "arrays of a struct and a struct by value reach a template kernel by name" do
      mod = Compiler.new.compile_with_cache(MATRIX, cache_dir: CACHE_DIR, name_expressions: ["kernel<float>", "kernel<double>"])
      n = 8
      [[Cumo::SFloat, "f", "kernel<float>", 1e-5], [Cumo::DFloat, "d", "kernel<double>", 1e-12]].each do |dtype, pack, name, tol|
        a = dtype.new(n, 4, 4).rand
        b = dtype.new(n, 4, 4).rand
        c = dtype.new(4, 4).rand
        out = dtype.new(n, 4, 4)
        mod.get_function(name).launch([a, b, c.to_a.flatten.pack("#{pack}*"), out], grid: 1, block: n)
        want = a.gemm(b) + c[:new, true, true]
        assert { (out - want).abs.max.to_f < tol }
      end
      assert_raise(DriverError) { mod.get_function("kernel<int>") }
    end

    def without_nvrtc
      compiler = Compiler.new
      compiler.define_singleton_method(:compile_using_nvrtc) { |*| raise "compiled again" }
      compiler
    end

    test "the mangled names come back from the cache with the cubin, and a broken file is recompiled" do
      dir = "#{CACHE_DIR}_round_trip"
      FileUtils.rm_rf(dir)
      first = Compiler.new.compile_with_cache(MATRIX, cache_dir: dir, name_expressions: ["kernel<float>", "kernel<float>"])
      files = Dir.glob(File.join(dir, "*_3.cubin"))
      assert_equal(1, files.size)
      assert_operator File.size(files[0]), :>, 1000
      again = without_nvrtc.compile_with_cache(MATRIX, cache_dir: dir, name_expressions: ["kernel<float>"])
      assert_equal(first.lowered_names, again.lowered_names)
      assert_nothing_raised { again.get_function("kernel<float>").launch([Cumo::SFloat.zeros(1, 4, 4), Cumo::SFloat.zeros(1, 4, 4), ([0.0] * 16).pack("f*"), Cumo::SFloat.zeros(1, 4, 4)], grid: 1, block: 1) }
      assert_raise(::RuntimeError) { without_nvrtc.compile_with_cache(MATRIX, cache_dir: dir, name_expressions: ["kernel<double>"]) }

      plain = Compiler.new.compile_with_cache(MATRIX, cache_dir: dir)
      assert_equal({}, plain.lowered_names)
      assert_raise(DriverError) { plain.get_function("kernel<float>") }
      assert_equal(1, Dir.glob(File.join(dir, "*_2.cubin")).size)
      assert_nothing_raised { without_nvrtc.compile_with_cache(MATRIX, cache_dir: dir) }

      [->(f) { File.truncate(f, File.size(f) / 2) }, ->(f) { File.write(f, "") }, ->(f) { File.write(f, "x" * 40) }].each do |spoil|
        spoil.call(files[0])
        assert_raise(::RuntimeError) { without_nvrtc.compile_with_cache(MATRIX, cache_dir: dir, name_expressions: ["kernel<float>"]) }
        fixed = Compiler.new.compile_with_cache(MATRIX, cache_dir: dir, name_expressions: ["kernel<float>"])
        assert_equal(first.lowered_names, fixed.lowered_names)
      end
      FileUtils.rm_rf(dir)
    end

    # CuPy's custom_struct/complex_struct: the layout a kernel reports is what
    # pack's @ offsets reproduce.
    test "a struct by value is packed at the offsets the device reports" do
      struct = "struct complex_struct { int4 a; char b; double c[2]; short1 d; unsigned long long int e[3]; };\n"
      layout = Module.new
      layout.load(Compiler.new.compile_using_nvrtc(struct + <<~CUDA))
        extern "C" __global__ void get_struct_layout(unsigned long long* itemsize, unsigned long long* sizes, unsigned long long* offsets) {
          const complex_struct* p = nullptr;
          itemsize[0] = sizeof(complex_struct);
          sizes[0] = sizeof(p->a); sizes[1] = sizeof(p->b); sizes[2] = sizeof(p->c); sizes[3] = sizeof(p->d); sizes[4] = sizeof(p->e);
          offsets[0] = (unsigned long long)&p->a; offsets[1] = (unsigned long long)&p->b; offsets[2] = (unsigned long long)&p->c;
          offsets[3] = (unsigned long long)&p->d; offsets[4] = (unsigned long long)&p->e;
        }
      CUDA
      itemsize = Cumo::UInt64.zeros(1)
      sizes = Cumo::UInt64.zeros(5)
      offsets = Cumo::UInt64.zeros(5)
      layout.get_function("get_struct_layout").launch([itemsize, sizes, offsets], grid: 1, block: 1)
      assert_equal([80], itemsize.to_a)
      assert_equal([16, 1, 16, 2, 24], sizes.to_a)
      off = offsets.to_a
      assert_equal([0, 16, 24, 40, 48], off)

      a = [1, 2, 3, 4]
      b = 5
      c = [1.5, 2.5]
      d = 7
      e = [10, 20, 30]
      bytes = [*a, b, *c, d, *e].pack("@#{off[0]}l4@#{off[1]}c@#{off[2]}d2@#{off[3]}s@#{off[4]}Q3@80")
      kernel = Module.new
      kernel.load(Compiler.new.compile_using_nvrtc(struct + <<~CUDA))
        extern "C" __global__ void test_kernel(const complex_struct s, double* out) {
          int i = threadIdx.x;
          double sum = 0.0;
          sum += s.a.x + s.a.y + s.a.z + s.a.w;
          sum += s.b;
          sum += s.c[0] + s.c[1];
          sum += s.d.x;
          sum += s.e[0] + s.e[1] + s.e[2];
          out[i] = i * sum;
        }
      CUDA
      out = Cumo::DFloat.zeros(4)
      kernel.get_function("test_kernel").launch([bytes, out], grid: 1, block: 4)
      want = (a.sum + b + c.sum + d + e.sum).to_f
      assert_equal([0, 1, 2, 3].map { |i| i * want }, out.to_a)
      layout.unload
      kernel.unload
    end
  end
end
