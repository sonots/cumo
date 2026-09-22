# frozen_string_literal: true

# Runs each ported example under Cumo and under the CuPy original, interleaved,
# and reports the times each one prints.
#
#   ruby bench/compare.rb [--rounds N] [--only NAME,NAME]
#
# CUPY_PYTHON points at a Python with cupy installed; CUPY_EXAMPLES at a
# checkout of cupy/examples.

require 'optparse'
require 'open3'

CUPY_PYTHON = ENV.fetch('CUPY_PYTHON', 'python3')
CUPY_EXAMPLES = ENV.fetch('CUPY_EXAMPLES', File.expand_path('~/prj/cupy/examples'))
CUMO_EXAMPLES = File.expand_path('..', __dir__)

# name => [cumo argv, cupy argv (relative to its example dir), metrics]
# A metric is [label, regexp] where the regexp captures one number.
CASES = {
  'cg' => {
    cumo: %w[cg/cg.rb],
    cupy: %w[cg/cg.py],
    env: { 'GPU' => '1' },
    metrics: [['CPU sec', / CPU\s*:\s+([\d.]+) sec/], ['GPU sec', / GPU\s*:\s+([\d.]+) sec/]]
  },
  'kmeans' => {
    cumo: %w[kmeans/kmeans.rb],
    cupy: %w[kmeans/kmeans.py],
    env: { 'GPU' => '1', 'MPLBACKEND' => 'Agg' },
    metrics: [['CPU sec', / CPU\s*:\s+([\d.]+) sec/], ['GPU sec', / GPU\s*:\s+([\d.]+) sec/]]
  },
  'kmeans-custom' => {
    cumo: %w[kmeans/kmeans.rb --use-custom-kernel],
    cupy: %w[kmeans/kmeans.py --use-custom-kernel],
    env: { 'GPU' => '1', 'MPLBACKEND' => 'Agg' },
    metrics: [['GPU sec', / GPU\s*:\s+([\d.]+) sec/]]
  },
  'gmm' => {
    cumo: %w[gmm/gmm.rb],
    cupy: %w[gmm/gmm.py],
    env: { 'GPU' => '1', 'MPLBACKEND' => 'Agg' },
    metrics: [['CPU sec', / CPU\s*:\s+([\d.]+) sec/], ['GPU sec', / GPU\s*:\s+([\d.]+) sec/]]
  },
  'black_scholes' => {
    cumo: %w[finance/black_scholes.rb],
    cupy: %w[finance/black_scholes.py],
    env: { 'GPU' => '1' },
    metrics: [
      ['CPU naive sec', /CPU \([^)]+\):\s+([\d.]+) sec/],
      ['GPU naive sec', /GPU \([^,]+, Naive[^)]*\):\s+([\d.]+) sec/],
      ['GPU kernel sec', /GPU \([^,]+, Elementwise[^)]*\):\s+([\d.]+) sec/]
    ]
  },
  'monte_carlo' => {
    cumo: %w[finance/monte_carlo.rb --n-threads-per-option 10000],
    cupy: %w[finance/monte_carlo.py --n-threads-per-option 10000],
    metrics: [['GPU sec', /Monte Carlo method\):\s+([\d.]+) sec/]]
  },
  'sgemm' => {
    cumo: %w[gemm/sgemm.rb --m 1024 --n 1024 --k 1024],
    cupy: %w[gemm/sgemm.py --m 1024 --n 1024 --k 1024],
    metrics: [['kernel ms', /hand written kernel time ([\d.]+) ms/], ['cuBLAS ms', /cuBLAS\s+time ([\d.]+) ms/]]
  }
}.freeze

def run(cmd, dir, env)
  out, err, status = Open3.capture3(env, *cmd, chdir: dir)
  [out + err, status.success?]
end

def cumo_cmd(argv)
  ['ruby', *argv]
end

def cupy_cmd(argv)
  [CUPY_PYTHON, *argv]
end

def extract(text, metrics)
  metrics.to_h do |label, re|
    values = text.scan(re).flatten.map(&:to_f)
    [label, values]
  end
end

options = { rounds: 3, only: nil }
OptionParser.new do |parser|
  parser.on('--rounds N', Integer) { |v| options[:rounds] = v }
  parser.on('--only LIST', Array) { |v| options[:only] = v }
end.parse!

cases = CASES.select { |name, _| options[:only].nil? || options[:only].include?(name) }
results = {}

cases.each do |name, spec|
  env = (spec[:env] || {}).dup
  cumo = cumo_cmd(spec[:cumo])
  cupy = cupy_cmd(spec[:cupy])
  cupy_dir = File.join(CUPY_EXAMPLES, File.dirname(spec[:cupy][0]))
  cupy_argv = [File.basename(spec[:cupy][0]), *spec[:cupy][1..]]
  cupy = cupy_cmd(cupy_argv)

  warn "== #{name}: warming up"
  run(cumo, CUMO_EXAMPLES, env)
  run(cupy, cupy_dir, env)

  samples = { 'cumo' => [], 'cupy' => [] }
  options[:rounds].times do |round|
    order = round.even? ? %w[cumo cupy] : %w[cupy cumo]
    order.each do |side|
      cmd, dir = side == 'cumo' ? [cumo, CUMO_EXAMPLES] : [cupy, cupy_dir]
      text, ok = run(cmd, dir, env)
      warn "   #{name} round #{round} #{side}: #{ok ? 'ok' : 'FAILED'}"
      warn text.lines.last(3).join.gsub(/^/, '     ') unless ok
      samples[side] << extract(text, spec[:metrics]) if ok
    end
  end
  results[name] = { spec: spec, samples: samples }
end

def summarize(values)
  return nil if values.empty?

  sorted = values.sort
  { min: sorted.first, med: sorted[sorted.size / 2], max: sorted.last, n: sorted.size }
end

puts
puts '| example | metric | cumo min/med/max | cupy min/med/max | cumo / cupy (median) |'
puts '| --- | --- | --- | --- | --- |'
results.each do |name, r|
  r[:spec][:metrics].each do |label, _|
    cumo_vals = r[:samples]['cumo'].flat_map { |h| h[label] }
    cupy_vals = r[:samples]['cupy'].flat_map { |h| h[label] }
    c = summarize(cumo_vals)
    p_ = summarize(cupy_vals)
    next if c.nil? || p_.nil?

    ratio = p_[:med].zero? ? '-' : format('%.2fx', c[:med] / p_[:med])
    fmt = ->(h) { format('%.4f / %.4f / %.4f', h[:min], h[:med], h[:max]) }
    puts "| #{name} | #{label} | #{fmt.call(c)} | #{fmt.call(p_)} | #{ratio} |"
  end
end
