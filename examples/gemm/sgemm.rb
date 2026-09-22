# frozen_string_literal: true

require 'optparse'

require 'cumo/narray'

require_relative 'utils'

SGEMM_FILE = File.join(__dir__, 'sgemm.cu')

# Cumo has no Fortran ordered array, and the bytes of a Fortran ordered [m, n]
# array are those of a C ordered [n, m] one, so a transposed copy is what the
# kernel is handed and what it writes into.
def as_fortran(a)
  a.transpose.dup
end

def sgemm(a, b,
          dim_x: 16, dim_y: 16, blk_m: 64, blk_n: 64, blk_k: 4,
          dim_xa: 64, dim_ya: 4, dim_xb: 4, dim_yb: 64)
  raise ArgumentError, 'A must be Cumo::SFloat' unless a.is_a?(Cumo::SFloat)
  raise ArgumentError, 'B must be Cumo::SFloat' unless b.is_a?(Cumo::SFloat)
  raise ArgumentError, 'thread block mismatch' unless dim_x * dim_y == dim_xa * dim_ya &&
                                                      dim_xa * dim_ya == dim_xb * dim_yb

  m, k = a.shape
  n = b.shape[1]

  # Inputs matrices need to be in Fortran order.
  a = as_fortran(a)
  b = as_fortran(b)

  c = Cumo::SFloat.new(n, m)

  config = {
    'DIM_X' => dim_x, 'DIM_Y' => dim_y,
    'BLK_M' => blk_m, 'BLK_N' => blk_n, 'BLK_K' => blk_k,
    'DIM_XA' => dim_xa, 'DIM_YA' => dim_ya,
    'DIM_XB' => dim_xb, 'DIM_YB' => dim_yb,
    'THR_M' => blk_m / dim_x, 'THR_N' => blk_n / dim_y
  }
  code = read_code(SGEMM_FILE, config)
  kern = Cumo::CUDA::Compiler.new.compile_with_cache(code).get_function('sgemm')

  grid = [(m.to_f / blk_m).ceil, (n.to_f / blk_n).ceil, 1]
  block = [dim_x, dim_y, 1]
  args = [[m].pack('l'), [n].pack('l'), [k].pack('l'), a, b, c]
  shared_mem = (blk_k * (blk_m + 1) * 4) + (blk_n * (blk_k + 1) * 4)
  kern.launch(args, grid: grid, block: block, shared_mem: shared_mem)
  c.transpose
end

def assert_array_almost_equal(actual, desired, decimal: 6)
  diff = (Cumo::DFloat.cast(actual) - Cumo::DFloat.cast(desired)).abs.max
  return if Float(diff) < 1.5 * (10**-decimal)

  raise "Arrays are not almost equal to #{decimal} decimals: largest difference #{Float(diff)}"
end

def main
  options = { gpu: 0, m: rand(1000...1500), n: rand(1000...1500), k: rand(500...3000) }
  OptionParser.new do |parser|
    parser.banner = 'SGEMM kernel call from Cumo'
    parser.on('-g', '--gpu ID', Integer, 'ID of GPU.') { |v| options[:gpu] = v }
    parser.on('--m M', Integer) { |v| options[:m] = v }
    parser.on('--n N', Integer) { |v| options[:n] = v }
    parser.on('--k K', Integer) { |v| options[:k] = v }
  end.parse!

  puts "m=#{options[:m]} n=#{options[:n]} k=#{options[:k]}"
  puts 'start benchmarking'
  puts ''

  kernel_times = nil
  cublas_times = nil
  Cumo::CUDA::Device.new(options[:gpu]).with do
    a = Cumo::SFloat.new(options[:m], options[:k]).rand(-1.0, 1.0)
    b = Cumo::SFloat.new(options[:k], options[:n]).rand(-1.0, 1.0)

    # check correctness
    assert_array_almost_equal(sgemm(a, b), a.dot(b), decimal: 3)

    # dry run
    3.times { sgemm(a, b) }
    kernel_times = benchmark(method(:sgemm), [a, b], 5)

    3.times { a.dot(b) }
    cublas_times = benchmark(->(x, y) { x.dot(y) }, [a, b], 5)
  end

  puts '=============================Result==============================='
  puts "hand written kernel time #{kernel_times.sum / kernel_times.size} ms"
  puts "cuBLAS              time #{cublas_times.sum / cublas_times.size} ms"
end

main if __FILE__ == $PROGRAM_NAME
