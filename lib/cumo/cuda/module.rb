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
      ObjectSpace.undefine_finalizer(self)
      Compiler.remove_module(self)
      Driver.cuModuleUnload(@ptr)
      @ptr = nil
    end

    def load_file(fname)
      loaded(Driver.cuModuleLoad(fname))
    end

    def load(cubin)
      loaded(Driver.cuModuleLoadData(cubin))
    end

    # A module nobody holds any more goes back to the driver. At exit the
    # context may be gone already, so a failure there is dropped.
    def self.unloader(ptr)
      proc do
        Driver.cuModuleUnload(ptr)
      rescue StandardError
        nil
      end
    end

    def loaded(ptr)
      unload
      @ptr = ptr
      ObjectSpace.define_finalizer(self, self.class.unloader(ptr))
      ptr
    end
    private :loaded

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
