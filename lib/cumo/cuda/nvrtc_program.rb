require_relative '../cuda'
require_relative 'compile_error'

module Cumo::CUDA
  class NVRTCProgram
    def initialize(src, name: "default_program", headers: [], include_names: [])
      @ptr = nil
      @src = src # should be UTF-8
      @name = name # should be UTF-8
      @ptr = NVRTC.nvrtcCreateProgram(src, name, headers, include_names)
    end

    def destroy
      NVRTC.nvrtcDestroyProgram(@ptr) if @ptr
    end

    def compile(options: [])
      begin
        NVRTC.nvrtcCompileProgram(@ptr, options)
        return NVRTC.nvrtcGetPTX(@ptr)
      rescue NVRTCError => e
        log = NVRTC.nvrtcGetProgramLog(@ptr)
        raise CompileError.new(log, @src, @name, options)
      end
    end
  end
end
