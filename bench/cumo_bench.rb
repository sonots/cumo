require 'cumo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

a = Cumo::Int32.new(10).seq(1)
b = Cumo::Int32.new(10).seq(10,10)
c = a + b
c.free

puts 'element-wise'
Benchmark.bm do |r|
  a = Cumo::Int32.new(10000).seq(1)
  b = Cumo::Int32.new(10000).seq(10,10)
  r.report('10**4') do
    NUM.times { (a + b).free }
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  a = Cumo::Int32.new(100000).seq(1)
  b = Cumo::Int32.new(100000).seq(10,10)
  r.report('10**5') do
    NUM.times { (a + b).free }
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  a = Cumo::Int32.new(1000000).seq(1)
  b = Cumo::Int32.new(1000000).seq(10,10)
  r.report('10**6') do
    NUM.times { (a + b).free }
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  a = Cumo::Int32.new(10000000).seq(1)
  b = Cumo::Int32.new(10000000).seq(10,10)
  r.report('10**7') do
    NUM.times { (a + b).free }
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end

  a = Cumo::Int32.new(100000000).seq(1)
  b = Cumo::Int32.new(100000000).seq(10,10)
  r.report('10**8') do
    NUM.times { (a + b).free }
    Cumo::CUDA::Runtime.cudaDeviceSynchronize
  end
end

# Tesla V100-SXM2...
#
# element-wise
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.000805)
# 10**5  0.000000   0.000000   0.000000 (  0.000959)
# 10**6  0.010000   0.000000   0.010000 (  0.004781)
# 10**7  0.030000   0.010000   0.040000 (  0.045121)
# 10**8  0.290000   0.160000   0.450000 (  0.443958
