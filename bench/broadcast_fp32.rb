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
# x.inplace + y          0.040000   0.000000   0.040000 (  0.039255)
# x + y                  0.140000   0.520000   0.660000 (  0.689061)
# x.inplace + 1.0        0.040000   0.000000   0.040000 (  0.040255)
# x.inplace + z          0.090000   0.000000   0.090000 (  0.083663)
# x.inplace - y          0.030000   0.010000   0.040000 (  0.039017)
# x.inplace - 1.0        0.050000   0.000000   0.050000 (  0.053187)
# x.inplace - z          0.080000   0.010000   0.090000 (  0.085909)
# x.inplace * y          0.030000   0.010000   0.040000 (  0.038583)
# x.inplace * 1.0        0.050000   0.010000   0.060000 (  0.051933)
# x.inplace * z          0.080000   0.010000   0.090000 (  0.084889)
# x.inplace / y          0.040000   0.000000   0.040000 (  0.038958)
# x.inplace / 1.0        0.040000   0.010000   0.050000 (  0.050266)
# x.inplace / z          0.080000   0.010000   0.090000 (  0.086685)
