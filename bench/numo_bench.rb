require 'numo/narray'
require 'benchmark'

# warm up
a = Numo::Int32.ones(10)
a * 2

[4, 5, 6, 7, 8].each do |digit|
  size = 10 ** digit
  started = Time.now
  a = Numo::Int32.ones(size)
  a * 2
  puts "10**#{digit}: #{(Time.now - started).to_f * 1000} msec"
end

# 10**4: 0.114593 msec
# 10**5: 0.919754 msec
# 10**6: 8.196666 msec
# 10**7: 80.11726800000001 msec
# 10**8: 721.842587 msec
