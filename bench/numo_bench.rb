require 'numo/narray'
require 'benchmark'

# warm up
a = Numo::Int32.ones(10)
a * 2

a = Numo::Int32.new(10000).seq(1)
b = Numo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report do
    100.times {
      1000.times { a + b }
      GC.start
    }
  end
end

# 10**4: 0.114593 msec
# 10**5: 0.919754 msec
# 10**6: 8.196666 msec
# 10**7: 80.11726800000001 msec
# 10**8: 721.842587 msec
