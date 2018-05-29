require 'numo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

# warm up
a = Numo::Int32.new(10).seq(1)
b = Numo::Int32.new(10).seq(10,10)
c = a + b

def elementwise
  puts 'element-wise'
  Benchmark.bm do |r|
    a = Numo::Int32.new(10000).seq(1)
    b = Numo::Int32.new(10000).seq(10,10)
    r.report('10**4') do
      NUM.times { (a + b) }
    end

    a = Numo::Int32.new(100000).seq(1)
    b = Numo::Int32.new(100000).seq(10,10)
    r.report('10**5') do
      NUM.times { (a + b) }
    end

    a = Numo::Int32.new(1000000).seq(1)
    b = Numo::Int32.new(1000000).seq(10,10)
    r.report('10**6') do
      NUM.times { (a + b) }
    end

    a = Numo::Int32.new(10000000).seq(1)
    b = Numo::Int32.new(10000000).seq(10,10)
    r.report('10**7') do
      NUM.times { (a + b) }
    end

    a = Numo::Int32.new(100000000).seq(1)
    b = Numo::Int32.new(100000000).seq(10,10)
    r.report('10**8') do
      NUM.times { (a + b) }
    end
  end
end

def reduction
  puts 'reduction'
  Benchmark.bm do |r|
    a = Numo::Int32.new(10000).seq(1)
    r.report('10**4') do
      NUM.times { (a.sum) }
    end

    a = Numo::Int32.new(100000).seq(1)
    r.report('10**5') do
      NUM.times { (a.sum) }
    end

    a = Numo::Int32.new(1000000).seq(1)
    r.report('10**6') do
      NUM.times { (a.sum) }
    end

    a = Numo::Int32.new(10000000).seq(1)
    r.report('10**7') do
      NUM.times { (a.sum) }
    end

    a = Numo::Int32.new(100000000).seq(1)
    r.report('10**8') do
      NUM.times { (a.sum) }
    end
  end
end

elementwise
reduction

# Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
#
# element-wise
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.002165)
# 10**5  0.000000   0.020000   0.020000 (  0.021530)
# 10**6  0.080000   0.040000   0.120000 (  0.117102)
# 10**7  0.980000   0.900000   1.880000 (  1.880362)
# 10**8  9.490000   8.580000  18.070000 ( 18.067418)
# reduction
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.000796)
# 10**5  0.010000   0.000000   0.010000 (  0.006547)
# 10**6  0.060000   0.000000   0.060000 (  0.063871)
# 10**7  0.640000   0.000000   0.640000 (  0.641754)
# 10**8  7.070000   0.000000   7.070000 (  7.071800)
