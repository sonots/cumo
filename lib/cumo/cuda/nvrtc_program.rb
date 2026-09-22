# frozen_string_literal: true

require_relative 'compile_error'

module Cumo::CUDA
  class NVRTCProgram
    # name_expressions: names such as "kernel<float>" whose mangled names
    # lowered_name answers once the program is compiled.
    def initialize(src, name: "default_program", headers: [], include_names: [], name_expressions: [])
      @ptr = nil
      @src = src # should be UTF-8
      @name = name # should be UTF-8
      name_expressions.each do |expr|
        raise TypeError, "a name expression is a String, got a #{expr.class}" unless expr.is_a?(String)
      end
      @ptr = NVRTC.nvrtcCreateProgram(src, name, headers, include_names)
      begin
        name_expressions.each { |expr| NVRTC.nvrtcAddNameExpression(@ptr, expr) }
      rescue Exception
        destroy
        raise
      end
    end

    def lowered_name(name_expression)
      NVRTC.nvrtcGetLoweredName(@ptr, name_expression)
    end

    def destroy
      return unless @ptr
      NVRTC.nvrtcDestroyProgram(@ptr)
      @ptr = nil
    end

    def compile(options: [])
      begin
        NVRTC.nvrtcCompileProgram(@ptr, options)
        return NVRTC.nvrtcGetPTX(@ptr)
      rescue NVRTCError
        log = NVRTC.nvrtcGetProgramLog(@ptr)
        raise CompileError.new(log, @src, @name, options)
      end
    end
  end
end
