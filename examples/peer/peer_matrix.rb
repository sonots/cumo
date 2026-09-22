# frozen_string_literal: true

require 'cumo/narray'

def main
  gpus = Cumo::CUDA::Runtime.cudaGetDeviceCount
  gpus.times do |peer_device|
    gpus.times do |device|
      next if peer_device == device

      flag = Cumo::CUDA::Device.new(device).can_access_peer?(peer_device)
      puts "Can access ##{peer_device} memory from ##{device}: #{flag}"
    end
  end
end

main if __FILE__ == $PROGRAM_NAME
