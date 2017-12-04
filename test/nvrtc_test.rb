require_relative "test_helper"

class NVRTCTest < Test::Unit::TestCase

  NVRTC = Numo::CUDA::NVRTC

  def test_create_program
    src = "__global__ void k() {}\n"
    name = "simple.cu"
    headers = []
    include_names = []
    assert_nothing_raised do
      prog = NVRTC.create_program(src, name, headers, include_names)
    end
  end

  def test_compile_program
    prog = test_create_program
    options = []
    assert_nothing_raised do
      NVRTC.compile_program(prog, options)
    end
  end

end
