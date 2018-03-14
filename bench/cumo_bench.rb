require 'cumo/narray'
require 'benchmark'

NUM = (ARGV.first || 100000).to_i

a = Cumo::Int32.new(10).seq(1)
b = Cumo::Int32.new(10).seq(10,10)
c = a + b
c.free

a = Cumo::Int32.new(10000).seq(1)
b = Cumo::Int32.new(10000).seq(10,10)
Benchmark.bm do |r|
  r.report(NUM) do
    NUM.times { (a + b).free }
  end
end

#              user     system      total        real
#    10000  0.100000   0.000000   0.100000 (  0.099510)
#   100000  0.880000   0.000000   0.880000 (  0.877870)
#  1000000  7.180000   0.000000   7.180000 (  7.178121)
# 10000000 83.140000   0.040000  83.180000 ( 83.179552)
