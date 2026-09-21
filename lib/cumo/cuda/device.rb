# frozen_string_literal: true

module Cumo::CUDA
  class Device
    attr_reader :id

    def self.get_currend_id
      Runtime.cudaGetDevice
    end

    def initialize(device_id = nil)
      if device_id
        @id = device_id
      else
        @id = Runtime.cudaGetDevice
      end
      @_device_stack = []
    end

    def use
      Runtime.cudaSetDevice(@id)
    end

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

    def synchronize
      Runtime.cudaDeviceSynchronize
    end

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
