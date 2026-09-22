# frozen_string_literal: true

# nsys profile --trace=cuda ruby stream/curand.rb
require 'cumo/narray'

# Cumo seeds one generator for the process rather than handing out generator
# objects, so this holds a seed and draws from that one. A seed given here is
# set once; drawing advances the sequence, and two of these cannot be
# interleaved and each keep its own place.
class RandomState
  def initialize(seed = nil)
    Cumo::NArray.srand(seed) unless seed.nil?
  end

  def lognormal(mean = 0.0, sigma = 1.0, size:)
    Cumo::NMath.exp(Cumo::DFloat.new(*size).rand_norm(mean, sigma))
  end
end

rand = RandomState.new

stream = Cumo::CUDA::Stream.new
y = stream.with { rand.lognormal(size: [1, 3]) }

stream = Cumo::CUDA::Stream.new
stream.use
y = rand.lognormal(size: [1, 3])
stream.synchronize
Cumo::CUDA::Stream.null.use

raise 'a lognormal draw is positive' unless Integer(y.gt(0).count_false).zero?
