require 'cumo/narray'
require 'benchmark'

a = Cumo::Int32.new(10).seq(1)
b = Cumo::Int32.new(10).seq(10,10)
c = a + b

a = Cumo::Int32.new(10000).seq(1)
b = Cumo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report do
    100000.times { c = a + b}
  end
end

