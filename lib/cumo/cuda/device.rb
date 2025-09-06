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
        Runtime.cudaSetDevice(@id) unless prev_id != @id
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
  end
end
