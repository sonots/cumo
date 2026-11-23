# frozen_string_literal: true

module Cumo::CUDA
  # Links PTX into a cubin with the CUDA driver, as Compiler does. With a
  # block, new yields the link and destroys it after.
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

    # Releases the link. Calling it again does nothing.
    def destroy
      return unless @ptr
      Driver.cuLinkDestroy(@ptr)
      @ptr = nil
    end

    # Adds PTX held in a String. name labels it in the messages of the driver.
    def add_ptr_data(data, name)
      Driver.cuLinkAddData(@ptr, Driver::CU_JIT_INPUT_PTX, data, name)
    end

    # Links what was added, and returns the cubin as a String.
    def complete
      Driver.cuLinkComplete(@ptr)
    end
  end
end
