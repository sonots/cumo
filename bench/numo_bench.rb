require 'numo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

# warm up
a = Numo::Float32.new(10).seq(1)
b = Numo::Float32.new(10).seq(10,10)
c = a + b

def elementwise
  puts 'element-wise'
  Benchmark.bm do |r|
    a = Numo::Float32.new(10000).seq(1)
    b = Numo::Float32.new(10000).seq(10,10)
    r.report('10**4') do
      NUM.times { (a + b) }
    end

    a = Numo::Float32.new(100000).seq(1)
    b = Numo::Float32.new(100000).seq(10,10)
    r.report('10**5') do
      NUM.times { (a + b) }
    end

    a = Numo::Float32.new(1000000).seq(1)
    b = Numo::Float32.new(1000000).seq(10,10)
    r.report('10**6') do
      NUM.times { (a + b) }
    end

    a = Numo::Float32.new(10000000).seq(1)
    b = Numo::Float32.new(10000000).seq(10,10)
    r.report('10**7') do
      NUM.times { (a + b) }
    end

    a = Numo::Float32.new(100000000).seq(1)
    b = Numo::Float32.new(100000000).seq(10,10)
    r.report('10**8') do
      NUM.times { (a + b) }
    end
  end
end

def reduction
  puts 'reduction'
  Benchmark.bm do |r|
    a = Numo::Float32.new(10000).seq(1)
    r.report('10**4') do
      NUM.times { (a.sum) }
    end

    a = Numo::Float32.new(100000).seq(1)
    r.report('10**5') do
      NUM.times { (a.sum) }
    end

    a = Numo::Float32.new(1000000).seq(1)
    r.report('10**6') do
      NUM.times { (a.sum) }
    end

    a = Numo::Float32.new(10000000).seq(1)
    r.report('10**7') do
      NUM.times { (a.sum) }
    end

    a = Numo::Float32.new(100000000).seq(1)
    r.report('10**8') do
      NUM.times { (a.sum) }
    end
  end
end

def dot
  num = 3
  puts 'dot'
  Benchmark.bm do |r|
    a = Numo::Float32.new(100,100).seq(1)
    b = Numo::Float32.new(100,100).seq(10,10)
    r.report('10**4') do
      num.times { a.dot(b) }
    end

    a = Numo::Float32.new(100,1000).seq(1)
    b = Numo::Float32.new(1000,100).seq(10,10)
    r.report('10**5') do
      num.times { a.dot(b) }
    end

    a = Numo::Float32.new(100,10000).seq(1)
    b = Numo::Float32.new(10000,100).seq(10,10)
    r.report('10**6') do
      num.times { a.dot(b) }
    end

    a = Numo::Float32.new(100,100000).seq(1)
    b = Numo::Float32.new(100000,100).seq(10,10)
    r.report('10**7') do
      num.times { a.dot(b) }
    end

    a = Numo::Float32.new(100,1000000).seq(1)
    b = Numo::Float32.new(1000000,100).seq(10,10)
    r.report('10**8') do
      num.times { a.dot(b) }
    end
  end
end

elementwise
reduction
dot

# Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
#
# element-wise
#        user     system      total        real
# 10**4  0.010000   0.000000   0.010000 (  0.002212)
# 10**5  0.000000   0.020000   0.020000 (  0.021604)
# 10**6  0.060000   0.060000   0.120000 (  0.120241)
# 10**7  0.980000   0.890000   1.870000 (  1.874592)
# 10**8  9.530000   8.520000  18.050000 ( 18.054087)
# reduction
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.001313)
# 10**5  0.010000   0.000000   0.010000 (  0.011400)
# 10**6  0.110000   0.000000   0.110000 (  0.111674)
# 10**7  1.120000   0.000000   1.120000 (  1.127018)
# 10**8 11.770000   0.010000  11.780000 ( 11.770858)
# dot
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.003935)
# 10**5  0.040000   0.000000   0.040000 (  0.037682)
# 10**6  0.380000   0.000000   0.380000 (  0.377312)
# 10**7  3.790000   0.000000   3.790000 (  3.792297)
# 10**8 38.820000   0.000000  38.820000 ( 38.816987)
