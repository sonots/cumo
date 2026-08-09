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
