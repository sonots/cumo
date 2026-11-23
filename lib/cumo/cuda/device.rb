# frozen_string_literal: true

module Cumo::CUDA
  # A CUDA device, by its id.
  class Device
    attr_reader :id

    # The id of the device the calling thread uses.
    def self.current_id
      Runtime.cudaGetDevice
    end

    # The device of that id, or the current one of the calling thread.
    def initialize(device_id = nil)
      if device_id
        @id = device_id
      else
        @id = Runtime.cudaGetDevice
      end
      @_device_stack = []
    end

    # Makes this the current device of the calling thread.
    def use
      Runtime.cudaSetDevice(@id)
    end

    # Runs the block with this as the current device of the calling thread,
    # restores the previous one after, and returns the value of the block.
    # @raise [RuntimeError] without a block.
    def with
      raise unless block_given?
      prev_id = Runtime.cudaGetDevice
      @_device_stack << prev_id
      begin
        Runtime.cudaSetDevice(@id) if prev_id != @id
        yield
      ensure
        prev_id = @_device_stack.pop
        Runtime.cudaSetDevice(prev_id)
      end
    end

    # Waits for everything queued on the current device of the calling
    # thread, which is this device only after use or inside with.
    def synchronize
      Runtime.cudaDeviceSynchronize
    end

    # The compute capability as a String of the major and the minor numbers,
    # such as "86" for 8.6.
    def compute_capability
      major = Runtime.cudaDeviceGetAttributes(75, @id)
      minor = Runtime.cudaDeviceGetAttributes(76, @id)
      "#{major}#{minor}"
    end

    # Whether this device can directly access the memory of another.
    def can_access_peer?(peer)
      peer = peer.id if peer.is_a?(Device)
      raise TypeError, "a peer is a device id or a Device, not a #{peer.class}" unless peer.is_a?(Integer)
      Runtime.cudaDeviceCanAccessPeer(@id, peer) != 0
    end
  end
end
