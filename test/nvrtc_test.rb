require File.join(__dir__, "../ext/numo/narray/narray")
require_relative "test_helper"

class NVRTCTest < Test::Unit::TestCase

  NVRTC = Numo::CUDA::NVRTC

  def test_create_program
    src = "__global__ void k() {}\n"
    name = "simple.cu"
    headers = []
    include_names = []
    prog = NVRTC.create_program(src, name, headers, include_names)
  end

  def test_compile_program
  end

end
