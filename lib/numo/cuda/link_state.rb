require_relative '../cuda'

module Numo::CUDA
  # CUDA link state.
  class LinkState
    def initialize
      @ptr = Driver.cuLinkCreate
      if block_given?
        begin
          yield(self)
        ensure
          destroy
        end
      end
    end

    def destroy
      return unless @ptr
      Driver.cuLinkDestroy(@ptr)
      @ptr = nil
    end

    def add_ptr_data(data, name)
      Driver.cuLinkAddData(@ptr, Driver::CU_JIT_INPUT_PTX, data, name)
    end

    def complete
      cubin = Driver.cuLinkComplete(@ptr)
    end
  end
end
