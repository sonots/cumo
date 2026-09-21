# frozen_string_literal: true

module Cumo::CUDA
  # A CUDA stream. Cumo launches every kernel and copy on the current stream
  # of the thread, which is the null stream until a Stream is used.
  #
  #   s = Cumo::CUDA::Stream.new
  #   s.with { c = a.gemm(b) }   # queued on s, and waited for on the way out
  #
  # Fibers share a thread's current stream.
  class Stream
    LIVE = ObjectSpace::WeakMap.new

    attr_reader :ptr

    def self.null
      @null ||= allocate.tap { |s| s.instance_variable_set(:@ptr, 0) }
    end

    # The Stream whose handle is the thread's current stream.
    def self.current
      ptr = Runtime.current_stream
      return null if ptr == 0
      LIVE[ptr] ||= allocate.tap { |s| s.instance_variable_set(:@ptr, ptr) }
    end

    # A handle may be given out again after a destroy, so a stream that is
    # still owned by a live object is not the one this finalizer was for.
    def self.finalizer(ptr)
      proc do
        unless LIVE[ptr]
          MemoryPool.free_all_blocks(ptr) rescue nil
          Runtime.cudaStreamDestroy(ptr) rescue nil
        end
      end
    end

    def initialize(non_blocking: false)
      flags = non_blocking ? Runtime::CUDA_STREAM_NON_BLOCKING : Runtime::CUDA_STREAM_DEFAULT
      @ptr = Runtime.cudaStreamCreateWithFlags(flags)
      LIVE[@ptr] = self
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    def null?
      @ptr == 0
    end

    # The handle of a stream that has not been destroyed.
    def handle
      raise ArgumentError, "the stream is destroyed" if @ptr.nil?
      @ptr
    end

    # Makes this the current stream of the thread until another is used.
    def use
      Runtime.current_stream = handle
      self
    end

    # Runs the block with this as the current stream, waits for everything
    # the block queued, and puts the previous stream back. The block's work
    # is ordered after what the previous stream had queued, and nothing the
    # block started is left running once it returns.
    def with
      before = Stream.current
      if before.handle != handle
        wait_event(before.record(Event.new(timing: false)))
      end
      Runtime.current_stream = handle
      begin
        yield self
      ensure
        begin
          synchronize unless null?
        ensure
          Runtime.current_stream = before.handle
        end
      end
    end

    def synchronize
      Runtime.cudaStreamSynchronize(handle)
      self
    end

    def done?
      Runtime.cudaStreamQuery(handle)
    end

    # Records an event after everything queued so far, and answers it.
    def record(event = nil)
      (event || Event.new).record(self)
    end

    # Makes everything queued after this call wait for the event.
    def wait_event(event)
      Runtime.cudaStreamWaitEvent(handle, event.handle)
      self
    end

    # Waits for the stream, hands its pooled memory back, and destroys it. A
    # stream that is current in a thread is refused, and stays usable.
    def destroy
      return if @ptr.nil? || null?
      Runtime.cudaStreamSynchronize(@ptr)
      Runtime.cudaStreamDestroy(@ptr)
      MemoryPool.free_all_blocks(@ptr)
      ObjectSpace.undefine_finalizer(self)
      LIVE.delete(@ptr) if LIVE.respond_to?(:delete)
      @ptr = nil
    end
  end
end
