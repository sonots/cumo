# frozen_string_literal: true

module Cumo::CUDA
  # CUDA kernel module.
  class Module
    def initialize
      @ptr = nil
      if block_given?
        begin
          yield(self)
        ensure
          unload
        end
      end
    end

    def unload
      return unless @ptr
      Driver.cuModuleUnload(@ptr)
      @ptr = nil
    end

    def load_file(fname)
      @ptr = Driver.cuModuleLoad(fname)
    end

    def load(cubin)
      @ptr = Driver.cuModuleLoadData(cubin)
    end

    def get_global_var(name)
      Driver.cuModuleGetGlobal(@ptr, name)
    end

    attr_accessor :lowered_names

    # Answers the kernel of that name as a Function: the name in the source
    # of a kernel inside extern "C", a name expression given at compile
    # time such as "kernel<float>", or a mangled name.
    def get_function(name)
      raise ArgumentError, "no module is loaded" unless @ptr
      lowered = (@lowered_names || {}).fetch(name, name)
      Function.new(self, Driver.cuModuleGetFunction(@ptr, lowered), name)
    end
  end
end
