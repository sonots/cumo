require 'numo/narray'
require 'benchmark'

NUM = (ARGV.first || 100000).to_i

# warm up
a = Numo::Int32.new(10).seq(1)
b = Numo::Int32.new(10).seq(10,10)
c = a + b

a = Numo::Int32.new(10000).seq(1)
b = Numo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report(NUM) do
    NUM.times { a + b }
  end
end

# 10**4: 0.114593 msec
# 10**5: 0.919754 msec
# 10**6: 8.196666 msec
# 10**7: 80.11726800000001 msec
# 10**8: 721.842587 msec
