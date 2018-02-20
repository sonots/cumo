require 'cumo/narray'
require 'benchmark'

# warm up
a = Cumo::Int32.ones(10)
a * 2

# Size
# 104 105 106 107 108
# NumPy [ms] 0.03 0.20 2.00 55.55 517.17
# CuPy [ms] 0.58 0.97 1.84 12.48 84.73

[4, 5, 6, 7, 8].each do |digit|
  size = 10 ** digit
  Cumo::CUDA::Runtime.cudaDeviceSynchronize
  started = Time.now
  a = Cumo::Int32.ones(size)
  a * 2
  Cumo::CUDA::Runtime.cudaDeviceSynchronize
  puts "10**#{digit}: #{(Time.now - started).to_f * 1000} msec"
end

# 10**4: 0.43667100000000003 msec
# 10**5: 0.35122000000000003 msec
# 10**6: 1.0603609999999999 msec
# 10**7: 7.784008999999999 msec
# 10**8: 74.664282 msec
