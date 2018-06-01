require 'numo/narray'
require 'benchmark'

NUM = (ARGV.first || 100).to_i

# warm up
a = Numo::Float32.new(10).seq(1)
b = Numo::Float32.new(10).seq(10,10)
c = a + b

def elementwise(num = nil)
  num ||= NUM
  puts "elementwise(#{num})"
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

def reduction(num = nil)
  num ||= NUM
  puts "reduction(#{num})"
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

def dot(num = nil)
  num ||= 1
  puts "dot(#{num})"
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
# elementwise(100)
#        user     system      total        real
# 10**4  0.010000   0.000000   0.010000 (  0.002368)
# 10**5  0.000000   0.020000   0.020000 (  0.024129)
# 10**6  0.080000   0.050000   0.130000 (  0.139918)
# 10**7  1.230000   1.020000   2.250000 (  2.251331)
# 10**8 10.090000   8.560000  18.650000 ( 18.646369)
# reduction(100)
#        user     system      total        real
# 10**4  0.000000   0.000000   0.000000 (  0.001360)
# 10**5  0.020000   0.000000   0.020000 (  0.011455)
# 10**6  0.110000   0.000000   0.110000 (  0.111708)
# 10**7  1.130000   0.000000   1.130000 (  1.137357)
# 10**8 11.830000   0.000000  11.830000 ( 11.832832)
# dot(1)
#        user     system      total        real
# 10**4  0.010000   0.000000   0.010000 (  0.001390)
# 10**5  0.010000   0.000000   0.010000 (  0.012563)
# 10**6  0.120000   0.010000   0.130000 (  0.125406)
# 10**7  1.270000   0.000000   1.270000 (  1.272804)
# 10**8 13.000000   0.000000  13.000000 ( 12.990586)
