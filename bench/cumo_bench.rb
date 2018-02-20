require 'cumo/narray'
require 'benchmark'

a = Cumo::Int32.new(10).seq(1)
b = Cumo::Int32.new(10).seq(10,10)
c = a + b

a = Cumo::Int32.new(10000).seq(1)
b = Cumo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report do
    100.times {
      1000.times { a + b }
      GC.start
    }
  end
end

# 10**4: 0.43667100000000003 msec
# 10**5: 0.35122000000000003 msec
# 10**6: 1.0603609999999999 msec
# 10**7: 7.784008999999999 msec
# 10**8: 74.664282 msec
