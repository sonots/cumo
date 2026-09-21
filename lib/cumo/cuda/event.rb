# frozen_string_literal: true

module Cumo::CUDA
  # A CUDA event: a point in a stream that can be waited for, and timed.
  #
  #   start = Cumo::CUDA::Event.new.record
  #   c = a.gemm(b)
  #   stop = Cumo::CUDA::Event.new.record
  #   stop.synchronize
  #   Cumo::CUDA.get_elapsed_time(start, stop)   # => milliseconds
  class Event
    LIVE = ObjectSpace::WeakMap.new

    attr_reader :ptr

    # A handle may be given out again after a destroy, so an event that is
    # still owned by a live object is not the one this finalizer was for.
    def self.finalizer(ptr)
      proc { Runtime.cudaEventDestroy(ptr) rescue nil unless LIVE[ptr] }
    end

    # blocking: makes synchronize sleep rather than spin; timing: false
    # makes a lighter event that cannot be timed.
    def initialize(blocking: false, timing: true)
      flags = Runtime::CUDA_EVENT_DEFAULT
      flags |= Runtime::CUDA_EVENT_BLOCKING_SYNC if blocking
      flags |= Runtime::CUDA_EVENT_DISABLE_TIMING unless timing
      @ptr = Runtime.cudaEventCreateWithFlags(flags)
      LIVE[@ptr] = self
      ObjectSpace.define_finalizer(self, self.class.finalizer(@ptr))
    end

    # The handle of an event that has not been destroyed.
    def handle
      raise ArgumentError, "the event is destroyed" if @ptr.nil?
      @ptr
    end

    # Records the event on a stream, the current one by default, after
    # everything queued on it so far.
    def record(stream = Stream.current)
      raise TypeError, "record takes a Stream, got a #{stream.class}" unless stream.is_a?(Stream)
      Runtime.cudaEventRecord(handle, stream.handle)
      self
    end

    def synchronize
      Runtime.cudaEventSynchronize(handle)
      self
    end

    def done?
      Runtime.cudaEventQuery(handle)
    end

    def destroy
      return if @ptr.nil?
      Runtime.cudaEventDestroy(@ptr)
      ObjectSpace.undefine_finalizer(self)
      @ptr = nil
    end
  end

  # The milliseconds between two recorded events.
  def self.get_elapsed_time(start, stop)
    Runtime.cudaEventElapsedTime(start.handle, stop.handle)
  end
end
