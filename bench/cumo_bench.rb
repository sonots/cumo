require 'cumo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

a = Cumo::Float32.new(10).seq(1)
b = Cumo::Float32.new(10).seq(10,10)
c = a + b
c.free

def elementwise
  puts 'element-wise'
  Benchmark.bm do |r|
    a = Cumo::Float32.new(10000).seq(1)
    b = Cumo::Float32.new(10000).seq(10,10)
    (a + b).free # warm up
    r.report('10**4') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000).seq(1)
    b = Cumo::Float32.new(100000).seq(10,10)
    (a + b).free # warm up
    r.report('10**5') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(1000000).seq(1)
    b = Cumo::Float32.new(1000000).seq(10,10)
    (a + b).free # warm up
    r.report('10**6') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(10000000).seq(1)
    b = Cumo::Float32.new(10000000).seq(10,10)
    (a + b).free # warm up
    r.report('10**7') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000000).seq(1)
    b = Cumo::Float32.new(100000000).seq(10,10)
    (a + b).free # warm up
    r.report('10**8') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end
  end
end

def reduction
  puts 'reduction'
  Benchmark.bm do |r|
    a = Cumo::Float32.new(10000).seq(1)
    r.report('10**4') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000).seq(1)
    r.report('10**5') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(1000000).seq(1)
    r.report('10**6') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(10000000).seq(1)
    r.report('10**7') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000000).seq(1)
    r.report('10**8') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end
  end
end

def dot
  num = 3
  puts 'dot'
  Benchmark.bm do |r|
    a = Cumo::Float32.new(100,100).seq(1)
    b = Cumo::Float32.new(100,100).seq(10,10)
    a.dot(b).free # warm up
    r.report('10**4') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,1000).seq(1)
    b = Cumo::Float32.new(1000,100).seq(10,10)
    a.dot(b).free # warm up
    r.report('10**5') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,10000).seq(1)
    b = Cumo::Float32.new(10000,100).seq(10,10)
    a.dot(b).free # warm up
    r.report('10**6') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,100000).seq(1)
    b = Cumo::Float32.new(100000,100).seq(10,10)
    a.dot(b).free # warm up
    r.report('10**7') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,1000000).seq(1)
    b = Cumo::Float32.new(1000000,100).seq(10,10)
    a.dot(b).free # warm up
    r.report('10**8') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end
  end
end

elementwise
reduction
dot

# Tesla V100-SXM2...
#
# element-wise
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.005769)
# 10**5  0.010000   0.000000   0.010000 (  0.006609)
# 10**6  0.000000   0.010000   0.010000 (  0.010313)
# 10**7  0.040000   0.010000   0.050000 (  0.050986)
# 10**8  0.310000   0.130000   0.440000 (  0.449699)
# reduction
#        user     system      total        real
# 10**4  0.010000   0.000000   0.010000 (  0.009484)
# 10**5  0.020000   0.010000   0.030000 (  0.022071)
# 10**6  0.100000   0.050000   0.150000 (  0.152070)
# 10**7  1.150000   0.600000   1.750000 (  1.754977)
# 10**8 11.720000   5.750000  17.470000 ( 17.470990)
# dot
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.000351)
# 10**5  0.000000   0.000000   0.000000 (  0.000838)
# 10**6  0.000000   0.000000   0.000000 (  0.002702)
# 10**7  0.020000   0.010000   0.030000 (  0.024650)
# 10**8  0.180000   0.060000   0.240000 (  0.245101)
