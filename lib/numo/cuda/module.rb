require_relative '../cuda'

module Numo::CUDA
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

    def get_function(name)
      # Function(name)
    end
  end
end

