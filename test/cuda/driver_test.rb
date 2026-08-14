# frozen_string_literal: true

require_relative "../test_helper"

module Cumo::CUDA
  class DriverTest < Test::Unit::TestCase
    SOURCE = <<~CUDA
      extern "C" {
        __device__ int cumo_test_global[4] = {11, 22, 33, 44};
      }
      __global__ void cumo_test_kernel() {}
    CUDA

    def test_cuModuleGetGlobal
      ptx = Compiler.new.compile_using_nvrtc(SOURCE)
      Module.new do |mod|
        mod.load(ptx)
        # the global lives in device memory, so reading it needs a copy back
        var = mod.get_global_var("cumo_test_global")
        assert { var.bytesize == 16 }
        assert { var.unpack("l<4") == [11, 22, 33, 44] }
      end
    end

    # cuLinkAddData used to pass the PTX blob as the `name` argument and ignore
    # the one it was given, so this also covers that the name reaches the driver
    # without upsetting it.
    def test_link_state_round_trip
      ptx = Compiler.new.compile_using_nvrtc(SOURCE)
      cubin = nil
      LinkState.new do |link|
        link.add_ptr_data(ptx, "cumo_test.ptx")
        cubin = link.complete
      end
      Module.new do |mod|
        mod.load(cubin)
        assert { mod.get_global_var("cumo_test_global").unpack("l<4") == [11, 22, 33, 44] }
      end
    end

    # Every one of these read RSTRING_PTR off whatever it was handed, so a
    # non-String argument reached the driver as a garbage pointer and segfaulted.
    NON_STRINGS = [[1, 2, 3], 123456, nil, { a: 1 }].freeze

    def test_string_arguments_are_type_checked
      NON_STRINGS.each do |bad|
        assert_raise(TypeError) { Driver.cuModuleLoad(bad) }
        assert_raise(TypeError) { Driver.cuModuleLoadData(bad) }
        assert_raise(TypeError) { Driver.cuModuleGetFunction(0, bad) }
        assert_raise(TypeError) { Driver.cuModuleGetGlobal(0, bad) }
      end
    end

    # Handles cross into Ruby as plain Integers, so a made-up one used to reach
    # the driver as a pointer and segfault.
    def test_unknown_handles_are_rejected
      assert_raise(ArgumentError) { Driver.cuLinkDestroy(1) }
      assert_raise(ArgumentError) { Driver.cuLinkComplete(1) }
      assert_raise(ArgumentError) { Driver.cuLinkAddFile(1, Driver::CU_JIT_INPUT_PTX, "k.ptx") }
      assert_raise(ArgumentError) { Driver.cuLinkAddData(1, Driver::CU_JIT_INPUT_PTX, "", "k.ptx") }
      assert_raise(ArgumentError) { Driver.cuModuleUnload(1) }
      assert_raise(ArgumentError) { Driver.cuModuleGetFunction(1, "f") }
      assert_raise(ArgumentError) { Driver.cuModuleGetGlobal(1, "g") }
    end

    def test_link_state_is_not_destroyed_twice
      state = Driver.cuLinkCreate
      Driver.cuLinkDestroy(state)
      assert_raise(ArgumentError) { Driver.cuLinkDestroy(state) }
      assert_raise(ArgumentError) { Driver.cuLinkComplete(state) }
    end

    def test_module_is_not_unloaded_twice
      ptx = Compiler.new.compile_using_nvrtc(SOURCE)
      hmod = Driver.cuModuleLoadData(ptx)
      Driver.cuModuleUnload(hmod)
      assert_raise(ArgumentError) { Driver.cuModuleUnload(hmod) }
      assert_raise(ArgumentError) { Driver.cuModuleGetGlobal(hmod, "cumo_test_global") }
    end

    # The context Init_cumo_cuda_driver creates is what lets this work: the
    # runtime API only makes its primary context once an array operation
    # happens. In-process the suite has always run one by now, so this needs a
    # fresh interpreter to mean anything.
    def test_driver_call_works_before_any_array_operation
      lib = File.expand_path("../../../lib", __FILE__)
      script = <<~RUBY
        require "cumo"
        state = Cumo::CUDA::Driver.cuLinkCreate
        Cumo::CUDA::Driver.cuLinkDestroy(state)
        print "ok"
      RUBY
      out = IO.popen([RbConfig.ruby, "-I#{lib}", "-e", script], err: %i[child out], &:read)
      assert_equal("ok", out)
    end

    def test_link_state_arguments_are_type_checked
      NON_STRINGS.each do |bad|
        LinkState.new do |link|
          state = link.instance_variable_get(:@ptr)
          assert_raise(TypeError) { Driver.cuLinkAddFile(state, Driver::CU_JIT_INPUT_PTX, bad) }
          assert_raise(TypeError) { Driver.cuLinkAddData(state, Driver::CU_JIT_INPUT_PTX, bad, "n.ptx") }
          assert_raise(TypeError) { Driver.cuLinkAddData(state, Driver::CU_JIT_INPUT_PTX, "", bad) }
        end
      end
    end
  end
end
