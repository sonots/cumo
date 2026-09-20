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

    # Answering nil here left the caller holding one, and finding out at
    # whatever it tried to do with it. What is missing is the launch rather than
    # the lookup: Driver.cuModuleGetFunction is bound, cuLaunchKernel is not,
    # and there is no object to put a CUfunction in.
    def get_function(name)
      raise NotImplementedError,
            "Cumo cannot launch a kernel it compiled: cuLaunchKernel is not bound"
    end
  end
end
