# frozen_string_literal: true

require 'benchmark'
require 'cumo/narray'

num_iteration = 100
Cumo::CUDA::Runtime.cudaDeviceSynchronize

Benchmark.bm 30 do |r|
  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum" do
    num_iteration.times do
      x.sum
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum(axis: 0)" do
    num_iteration.times do
      x.sum(axis: 0)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum(axis: 1)" do
    num_iteration.times do
      x.sum(axis: 1)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum(keepdims: true)" do
    num_iteration.times do
      x.sum(keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum(axis: 0, keepdims: true)" do
    num_iteration.times do
      x.sum(axis: 0, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.sum(axis: 1, keepdims: true)" do
    num_iteration.times do
      x.sum(axis: 1, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max" do
    num_iteration.times do
      x.max
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max(axis: 0)" do
    num_iteration.times do
      x.max(axis: 0)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max(axis: 1)" do
    num_iteration.times do
      x.max(axis: 1)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max(keepdims: true)" do
    num_iteration.times do
      x.max(keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max(axis: 0, keepdims: true)" do
    num_iteration.times do
      x.max(axis: 0, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500, 500])
  r.report "x.max(axis: 1, keepdims: true)" do
    num_iteration.times do
      x.max(axis: 1, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end
end

#                                      user     system      total        real
# x.sum                            0.170000   0.020000   0.190000 (  0.188164)
# x.sum(axis: 0)                   0.070000   0.010000   0.080000 (  0.081719)
# x.sum(axis: 1)                   0.060000   0.020000   0.080000 (  0.080435)
# x.sum(keepdims: true)            0.120000   0.040000   0.160000 (  0.153970)
# x.sum(axis: 0, keepdims: true)   0.070000   0.010000   0.080000 (  0.083349)
# x.sum(axis: 1, keepdims: true)   0.080000   0.010000   0.090000 (  0.083299)
# x.max                            0.140000   0.020000   0.160000 (  0.158882)
# x.max(axis: 0)                   0.080000   0.000000   0.080000 (  0.081502)
# x.max(axis: 1)                   0.080000   0.000000   0.080000 (  0.080473)
# x.max(keepdims: true)            0.140000   0.020000   0.160000 (  0.159530)
# x.max(axis: 0, keepdims: true)   0.070000   0.020000   0.090000 (  0.083434)
# x.max(axis: 1, keepdims: true)   0.080000   0.000000   0.080000 (  0.083299)
