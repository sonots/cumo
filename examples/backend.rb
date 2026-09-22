# frozen_string_literal: true

# Numo is always loaded: it is the CPU side of every comparison, and the
# side the data is generated on.
require 'numo/narray'
HM = Numo

# Numo's dot only reaches BLAS when Numo::Linalg is defined; without it cg
# and gmm run an order of magnitude slower on the CPU side.
begin
  require 'numo/linalg'
rescue LoadError
  nil
end

# GPU=1 selects Cumo for the GPU side, as narray-llm does. Without it the
# examples run their CPU side only, so a machine without CUDA can run them.
if ENV['GPU'].to_s =~ /\A(1|on|true)\z/i
  require 'cumo/narray'
  XM = Cumo
else
  XM = Numo
end

module Backend
  module_function

  def gpu?
    XM.name == 'Cumo'
  end

  # cupy.get_array_module: answers the module the array belongs to.
  def array_module(a)
    gpu? && a.is_a?(Cumo::NArray) ? Cumo : Numo
  end

  def synchronize
    Cumo::CUDA::Runtime.cudaDeviceSynchronize if gpu?
  end

  # cupy.cuda.Device(id): selects the device for the block on the GPU side,
  # and just runs the block on the CPU side.
  def with_device(id)
    return yield unless gpu?

    Cumo::CUDA::Device.new(id).with { yield }
  end

  # cupy.asarray / ndarray.get: moves an array to the XM side or back to Numo,
  # keeping its class (SFloat stays SFloat).
  def to_device(a)
    same_class(XM, a).cast(a)
  end

  def to_host(a)
    same_class(HM, a).cast(a)
  end

  def same_class(mod, a)
    mod.const_get(a.class.name.split('::').last)
  end
end
