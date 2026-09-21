# frozen_string_literal: true

module Cumo::CUDA
  # A kernel of a loaded Module, ready to launch.
  class Function
    attr_reader :name

    def initialize(mod, ptr, name)
      @mod = mod
      @ptr = ptr
      @name = name
    end

    # Launches the kernel on the current stream, the one Cumo's own kernels run
    # on, so it runs after the operations issued before it and before the
    # ones issued after.
    #
    # An NArray argument hands over its device pointer, so the kernel sees the
    # elements themselves; it has to be contiguous. An Integer is passed as a
    # long long and a Float as a double. Anything else goes as packed bytes:
    # [n].pack("l") is an int, [x].pack("f") a float, and a packed struct is
    # a struct passed by value.
    #
    # grid and block take one to three sizes each; shared_mem is the dynamic
    # shared memory in bytes; stream: is a Stream to launch on instead of the
    # current one.
    def launch(args, grid:, block:, shared_mem: 0, stream: nil)
      unless shared_mem.is_a?(Integer) && shared_mem >= 0
        raise ArgumentError, "shared_mem is a byte count, got #{shared_mem.inspect}"
      end
      raise TypeError, "stream: takes a Stream, got a #{stream.class}" unless stream.nil? || stream.is_a?(Stream)
      Driver.cuLaunchKernel(@ptr, *dims(grid, "grid"), *dims(block, "block"), shared_mem, stream&.handle, args)
    end

    private

    def dims(v, what)
      d = v.is_a?(Integer) ? [v] : v
      unless d.is_a?(Array) && (1..3).cover?(d.size) && d.all? { |x| x.is_a?(Integer) && x >= 1 }
        raise ArgumentError, "#{what} takes one to three sizes of at least 1, got #{v.inspect}"
      end
      d + [1] * (3 - d.size)
    end
  end
end
