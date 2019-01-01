require 'cumo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

# warm up
a = Cumo::Float32.new(10).seq(1)
b = Cumo::Float32.new(10).seq(10,10)
c = a + b
c.free

def elementwise(num = nil)
  num ||= NUM
  puts "elementwise(#{num})"
  Benchmark.bm do |r|
    a = Cumo::Float32.new(10000).seq(1)
    b = Cumo::Float32.new(10000).seq(10,10)
    (a + b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**4') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000).seq(1)
    b = Cumo::Float32.new(100000).seq(10,10)
    (a + b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**5') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(1000000).seq(1)
    b = Cumo::Float32.new(1000000).seq(10,10)
    (a + b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**6') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(10000000).seq(1)
    b = Cumo::Float32.new(10000000).seq(10,10)
    (a + b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**7') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000000).seq(1)
    b = Cumo::Float32.new(100000000).seq(10,10)
    (a + b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**8') do
      NUM.times do
        (a + b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end
  end
end

def reduction(num = nil)
  num ||= NUM
  puts "reduction(#{num})"
  Benchmark.bm do |r|
    a = Cumo::Float32.new(10000).seq(1)
    (a.sum).free  # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**4') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000).seq(1)
    (a.sum).free  # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**5') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(1000000).seq(1)
    (a.sum).free  # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**6') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(10000000).seq(1)
    (a.sum).free  # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**7') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100000000).seq(1)
    (a.sum).free  # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**8') do
      NUM.times do
        (a.sum).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end
  end
end

def dot(num = nil)
  num ||= 1
  puts "dot(#{num})"
  Benchmark.bm do |r|
    a = Cumo::Float32.new(100,100).seq(1)
    b = Cumo::Float32.new(100,100).seq(10,10)
    a.dot(b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**4') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,1000).seq(1)
    b = Cumo::Float32.new(1000,100).seq(10,10)
    a.dot(b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**5') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,10000).seq(1)
    b = Cumo::Float32.new(10000,100).seq(10,10)
    a.dot(b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**6') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,100000).seq(1)
    b = Cumo::Float32.new(100000,100).seq(10,10)
    a.dot(b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
    r.report('10**7') do
      num.times do
        a.dot(b).free
        Cumo::CUDA::Runtime.cudaDeviceSynchronize
      end
    end

    a = Cumo::Float32.new(100,1000000).seq(1)
    b = Cumo::Float32.new(1000000,100).seq(10,10)
    a.dot(b).free # warm up
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
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
# element-wise(100)
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.006332)
# 10**5  0.000000   0.000000   0.000000 (  0.006280)
# 10**6  0.010000   0.000000   0.010000 (  0.008123)
# 10**7  0.000000   0.010000   0.010000 (  0.022176)
# 10**8  0.100000   0.050000   0.150000 (  0.151999)
# reduction(100)
#        user     system      total        real
# 10**4  0.010000   0.000000   0.010000 (  0.009735)
# 10**5  0.010000   0.010000   0.020000 (  0.022882)
# 10**6  0.110000   0.050000   0.160000 (  0.154641)
# 10**7  1.220000   0.590000   1.810000 (  1.805643)
# 10**8 11.840000   6.110000  17.950000 ( 17.946511)
# dot(1)
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.000206)
# 10**5  0.000000   0.000000   0.000000 (  0.000195)
# 10**6  0.000000   0.000000   0.000000 (  0.000239)
# 10**7  0.000000   0.000000   0.000000 (  0.000719)
# 10**8  0.010000   0.000000   0.010000 (  0.004636)
