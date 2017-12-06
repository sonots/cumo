require_relative '../cuda'
require_relative 'compile_error'

module Numo::CUDA
  class NVRTCProgram
    def initialize(src, name: "default_program", headers: [], include_names: [])
      @ptr = nil
      @src = src # should be UTF-8
      @name = name # should be UTF-8
      @ptr = NVRTC.create_program(src, name, headers, include_names)
    end

    def destroy
      NVRTC.destroy_program(@ptr) if @ptr
    end

    def compile(options: [])
      begin
        NVRTC.compile_program(@ptr, options)
        return NVRTC.get_ptx(@ptr)
      rescue NVRTCError => e
        log = NVRTC.get_program_log(@ptr)
        raise CompileError.new(log, @src, @name, options)
      end
    end
  end
end
