require 'benchmark'
require 'cumo/narray'

num_iteration = 100
Cumo::CUDA::Runtime.cudaDeviceSynchronize

Benchmark.bm 30 do |r|
  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum" do
    num_iteration.times do
      x.sum
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum(axis: 0)" do
    num_iteration.times do
      x.sum(axis: 0)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum(axis: 1)" do
    num_iteration.times do
      x.sum(axis: 1)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum(keepdims: true)" do
    num_iteration.times do
      x.sum(keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum(axis: 0, keepdims: true)" do
    num_iteration.times do
      x.sum(axis: 0, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.sum(axis: 1, keepdims: true)" do
    num_iteration.times do
      x.sum(axis: 1, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max" do
    num_iteration.times do
      x.max
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max(axis: 0)" do
    num_iteration.times do
      x.max(axis: 0)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max(axis: 1)" do
    num_iteration.times do
      x.max(axis: 1)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max(keepdims: true)" do
    num_iteration.times do
      x.max(keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max(axis: 0, keepdims: true)" do
    num_iteration.times do
      x.max(axis: 0, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([500,500])
  r.report "x.max(axis: 1, keepdims: true)" do
    num_iteration.times do
      x.max(axis: 1, keepdims: true)
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end
end

#                                      user     system      total        real
# x.sum                            1.250000   0.150000   1.400000 (  1.401491)
# x.sum(axis: 0)                   0.640000   0.070000   0.710000 (  0.717888)
# x.sum(axis: 1)                   1.200000   0.410000   1.610000 (  1.616371)
# x.sum(keepdims: true)            1.150000   0.220000   1.370000 (  1.371619)
# x.sum(axis: 0, keepdims: true)   0.610000   0.100000   0.710000 (  0.701329)
# x.sum(axis: 1, keepdims: true)   1.230000   0.360000   1.590000 (  1.592216)
# x.max                            1.450000   0.200000   1.650000 (  1.644903)
# x.max(axis: 0)                   0.770000   0.150000   0.920000 (  0.927981)
# x.max(axis: 1)                   1.380000   0.500000   1.880000 (  1.876824)
# x.max(keepdims: true)            1.450000   0.230000   1.680000 (  1.674981)
# x.max(axis: 0, keepdims: true)   0.760000   0.170000   0.930000 (  0.936248)
# x.max(axis: 1, keepdims: true)   1.400000   0.470000   1.870000 (  1.877935)
