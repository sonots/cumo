# frozen_string_literal: true

module Cumo::CUDA
  # Page-locked host memory, which a copy to or from the device can be
  # asynchronous with. NArray#set and NArray#get copy through it.
  #
  #   pinned = Cumo::CUDA::PinnedMemory.new(a.byte_size)
  #   s.with { a.get(pinned) }        # queued on s, complete when with returns
  #   Cumo::SFloat.from_binary(pinned.read, a.shape)
  #
  # A copy in flight keeps the array alive and is waited for by read, write,
  # free and the next copy, so the bytes read are the copy's.
  class PinnedMemory
    LIVE = ObjectSpace::WeakMap.new

    attr_reader :bytesize

    # A buffer holding the bytes of a String.
    def self.from_binary(bytes)
      new(bytes.bytesize).write(bytes)
    end

    # The state is shared with the finalizer, so a copy still in flight when
    # the buffer is dropped is waited for before the memory goes.
    def self.finalizer(state)
      proc do
        ptr, event = state
        event.synchronize rescue nil if event
        Runtime.cudaFreeHost(ptr) rescue nil unless LIVE[ptr]
      end
    end

    def initialize(bytesize, flags: Runtime::CUDA_HOST_ALLOC_DEFAULT)
      @ptr = Runtime.cudaHostAlloc(bytesize, flags)
      @bytesize = bytesize
      @state = [@ptr, nil]
      @busy_with = nil
      LIVE[@ptr] = self
      ObjectSpace.define_finalizer(self, self.class.finalizer(@state))
    end

    # The handle of a buffer that has not been freed.
    def handle
      raise ArgumentError, "the pinned host buffer is freed" if @ptr.nil?
      @ptr
    end

    # Waits for the copy in flight, if any.
    def synchronize
      @state[1]&.synchronize
      @state[1] = nil
      @busy_with = nil
      self
    end

    # The bytes, or length of them from offset, as a String.
    def read(length = nil, offset = 0)
      synchronize
      Runtime.pinned_read(handle, offset, length || @bytesize - offset)
    end
    alias to_binary read

    # Writes the bytes of a String at offset.
    def write(bytes, offset = 0)
      synchronize
      Runtime.pinned_write(handle, offset, bytes)
      self
    end

    def free
      return if @ptr.nil?
      synchronize
      Runtime.cudaFreeHost(@ptr)
      ObjectSpace.undefine_finalizer(self)
      @ptr = nil
    end

    # Queues a copy between the buffer and an array on a stream, the current
    # one when nil, and records where it ends. NArray#set and #get call this.
    def copy(narray, direction, stream)
      raise TypeError, "stream: takes a Stream, got a #{stream.class}" unless stream.nil? || stream.is_a?(Stream)
      synchronize
      if direction == :to_narray
        Runtime.memcpy_pinned_to_narray(handle, narray, stream&.handle)
      else
        Runtime.memcpy_narray_to_pinned(narray, handle, stream&.handle)
      end
      @busy_with = narray
      @state[1] = Event.new(timing: false).record(stream || Stream.current)
      self
    end
  end
end
