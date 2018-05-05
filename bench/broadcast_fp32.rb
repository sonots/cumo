require 'benchmark'
require 'cumo/narray'

num_iteration = 1000

Benchmark.bm 20 do |r|
  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace + y" do
    num_iteration.times do
      x.inplace + y
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x + y" do
    num_iteration.times do
      (x + y).free
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace + 1.0" do
    num_iteration.times do
      x.inplace + 1.0
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  z = Cumo::SFloat.ones([1000,1])
  r.report "x.inplace + z" do
    num_iteration.times do
      x.inplace + z
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace - y" do
    num_iteration.times do
      x.inplace - y
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace - 1.0" do
    num_iteration.times do
      x.inplace - 1.0
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  z = Cumo::SFloat.ones([1000,1])
  r.report "x.inplace - z" do
    num_iteration.times do
      x.inplace - z
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace * y" do
    num_iteration.times do
      x.inplace * y
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace * 1.0" do
    num_iteration.times do
      x.inplace * 1.0
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  z = Cumo::SFloat.ones([1000,1])
  r.report "x.inplace * z" do
    num_iteration.times do
      x.inplace * z
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace / y" do
    num_iteration.times do
      x.inplace / y
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  y = Cumo::SFloat.ones([1000,784])
  r.report "x.inplace / 1.0" do
    num_iteration.times do
      x.inplace / 1.0
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  x = Cumo::SFloat.ones([1000,784])
  z = Cumo::SFloat.ones([1000,1])
  r.report "x.inplace / z" do
    num_iteration.times do
      x.inplace / z
    end
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end
end

#                            user     system      total        real
# x.inplace + y          0.040000   0.000000   0.040000 (  0.038603)
# x + y                  0.080000   0.570000   0.650000 (  0.681606)
# x.inplace + 1.0        0.040000   0.010000   0.050000 (  0.037634)
# x.inplace + z          4.220000   0.020000   4.240000 (  4.240187)
# x.inplace - y          0.040000   0.000000   0.040000 (  0.038607)
# x.inplace - 1.0        0.040000   0.000000   0.040000 (  0.046983)
# x.inplace - z          4.170000   0.010000   4.180000 (  4.178762)
# x.inplace * y          0.030000   0.000000   0.030000 (  0.037780)
# x.inplace * 1.0        0.050000   0.010000   0.060000 (  0.047114)
# x.inplace * z          4.240000   0.000000   4.240000 (  4.239317)
# x.inplace / y          0.040000   0.000000   0.040000 (  0.038596)
# x.inplace / 1.0        0.040000   0.010000   0.050000 (  0.048086)
# x.inplace / z          4.420000   0.020000   4.440000 (  4.436268)

