require 'numo/narray'
require 'benchmark'

a = Numo::Int32.new(10).seq(1)
b = Numo::Int32.new(10).seq(10,10)
c = a + b

a = Numo::Int32.new(10000).seq(1)
b = Numo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report do
    100000.times { c = a + b}
  end
end
