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

    # Answers the kernel of that name as a Function. The name is the one in
    # the source, so a kernel outside extern "C" has to be asked for by its
    # mangled name.
    def get_function(name)
      raise ArgumentError, "no module is loaded" unless @ptr
      Function.new(self, Driver.cuModuleGetFunction(@ptr, name), name)
    end
  end
end
