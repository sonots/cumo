#!/usr/bin/env ruby
# frozen_string_literal: true

# ---------------------------------------------------------------------------
# Forward pass of a transformer block (GPT-2 style, pre-LayerNorm), timed on
# Numo and on Cumo.
#
#   ruby transformer_bench.rb                        # CPU: numo-narray (dot without BLAS)
#   LINALG=1 ruby transformer_bench.rb               # CPU: numo-linalg (BLAS) for GEMM
#   GPU=1 ruby transformer_bench.rb                  # GPU: cumo (cuBLAS)
#
#   # the shape of GPT-2 124M, for the GPU
#   GPU=1 T=1024 C=768 NH=12 LAYERS=12 ruby transformer_bench.rb
#
#   DETAIL=0 ...   drop the per-section synchronization and time the whole
#                  pipeline instead
#   HEAD_COPY=0 ...pass the head split to dot as a view rather than a copy, to
#                  see what Cumo accepts
#
# What this measures is the ratio of GEMM to everything else. GEMM reaches
# cuBLAS and is fast, while LayerNorm, softmax and GELU have no fused kernel in
# Cumo and run through a chain of temporaries, so the point is how much that
# bandwidth-bound remainder holds the block back.
# ---------------------------------------------------------------------------

GPU = !%w[0 false].include?(ENV['GPU'].to_s.downcase) && !ENV['GPU'].to_s.empty?

if GPU
  require 'cumo/narray'
  XM = Cumo
else
  require 'numo/narray'
  XM = Numo
end

# NArray#dot in numo-narray alone is a plain implementation that does not use
# BLAS, so a fair CPU GEMM needs numo-linalg.
USE_LINALG =
  if !GPU && ENV['LINALG'] == '1'
    begin
      require 'numo/linalg'
      Numo::Linalg.respond_to?(:matmul)
    rescue LoadError
      warn 'numo-linalg is not available, falling back to NArray#dot'
      false
    end
  else
    false
  end

T      = (ENV['T']      || 512).to_i   # sequence length
C      = (ENV['C']      || 1024).to_i   # embedding width
NH     = (ENV['NH']     || 8).to_i     # number of heads
LAYERS = (ENV['LAYERS'] || 5).to_i     # blocks to stack, all sharing one set of weights
ITER   = (ENV['ITER']   || 3).to_i
DETAIL = ENV['DETAIL'] != '0'
HEAD_COPY = ENV['HEAD_COPY'] != '0'

raise "C(#{C}) has to divide by NH(#{NH})" unless (C % NH).zero?

HS = C / NH
TIMES = Hash.new(0.0)

# --- helpers ---------------------------------------------------------------

def sync
  XM::CUDA::Runtime.cudaDeviceSynchronize if GPU
end

def scalar(v)
  return v if v.is_a?(Numeric)

  v.respond_to?(:extract_cpu) ? v.extract_cpu : v.extract
end

def timed(key)
  return yield unless DETAIL

  t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
  r = yield
  sync # without this Cumo only reports how long the launches took to queue
  TIMES[key] += Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
  r
end

def matmul(a, b)
  USE_LINALG ? Numo::Linalg.matmul(a, b) : a.dot(b)
end

# Copies a view (a transpose or a column slice) into contiguous memory, since
# not every implementation takes a non-contiguous array in dot.
def contiguous(a)
  out = XM::SFloat.zeros(*a.shape)
  out.store(a)
  out
end

def head(m, range)
  v = m[true, range]
  HEAD_COPY ? contiguous(v) : v
end

def head_t(m, range)
  v = m[true, range].transpose
  HEAD_COPY ? contiguous(v) : v
end

# --- layers ----------------------------------------------------------------

def layernorm(x, gain, bias)
  rows = x.shape[0]
  mu = x.mean(axis: 1).reshape(rows, 1)
  xc = x - mu
  var = (xc * xc).mean(axis: 1).reshape(rows, 1)
  xc / XM::NMath.sqrt(var + 1.0e-5) * gain + bias
end

# Softmax along the rows, with the usual subtract-the-max for stability.
def softmax_rows(a)
  rows = a.shape[0]
  m = a.max(axis: 1).reshape(rows, 1)
  e = XM::NMath.exp(a - m)
  e / e.sum(axis: 1).reshape(rows, 1)
end

def gelu(x)
  # the tanh approximation, as GPT-2 uses
  0.5 * x * (1.0 + XM::NMath.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
end

# --- transformer block -----------------------------------------------------

def block(x, w, mask)
  scale = 1.0 / Math.sqrt(HS)

  # --- attention ---
  x1 = timed(:layernorm) { layernorm(x, w[:g1], w[:b1]) }
  qkv = timed(:gemm_qkv) { matmul(x1, w[:w_qkv]) }
  qkv = timed(:bias_add) { qkv + w[:b_qkv] }

  q = qkv[true, 0...C]
  k = qkv[true, C...(2 * C)]
  v = qkv[true, (2 * C)...(3 * C)]

  attn = XM::SFloat.zeros(T, C)
  NH.times do |h|
    r = (h * HS)...((h + 1) * HS)
    qh = timed(:head_split) { head(q, r) }
    kt = timed(:head_split) { head_t(k, r) }
    vh = timed(:head_split) { head(v, r) }

    s = timed(:gemm_attn) { matmul(qh, kt) }           # [T, T]
    s = timed(:mask_scale) { s * scale + mask }
    p = timed(:softmax) { softmax_rows(s) }
    o = timed(:gemm_attn) { matmul(p, vh) }            # [T, HS]
    timed(:head_merge) { attn[true, r] = o }
  end

  proj = timed(:gemm_proj) { matmul(attn, w[:w_proj]) }
  x = timed(:residual) { x + (proj + w[:b_proj]) }

  # --- MLP ---
  x2 = timed(:layernorm) { layernorm(x, w[:g2], w[:b2]) }
  h1 = timed(:gemm_fc) { matmul(x2, w[:w_fc]) }
  h1 = timed(:gelu) { gelu(h1 + w[:b_fc]) }
  h2 = timed(:gemm_fc) { matmul(h1, w[:w_fcproj]) }
  timed(:residual) { x + (h2 + w[:b_fcproj]) }
end

def build_weights
  sd = 0.02
  {
    g1: XM::SFloat.ones(C),
    b1: XM::SFloat.zeros(C),
    g2: XM::SFloat.ones(C),
    b2: XM::SFloat.zeros(C),
    w_qkv: XM::SFloat.new(C, 3 * C).rand * (2 * sd) - sd,
    b_qkv: XM::SFloat.zeros(3 * C),
    w_proj: XM::SFloat.new(C, C).rand * (2 * sd) - sd,
    b_proj: XM::SFloat.zeros(C),
    w_fc: XM::SFloat.new(C, 4 * C).rand * (2 * sd) - sd,
    b_fc: XM::SFloat.zeros(4 * C),
    w_fcproj: XM::SFloat.new(4 * C, C).rand * (2 * sd) - sd,
    b_fcproj: XM::SFloat.zeros(C)
  }
end

# Causal mask: -1e9 above the diagonal, built with arithmetic rather than Bit.
def causal_mask
  i = XM::SFloat.new(T, 1).seq
  j = XM::SFloat.new(1, T).seq
  d = (XM::SFloat.zeros(T, T) + j) - i        # j - i
  -1.0e9 * d.clip(0.0, 1.0).ceil              # 1.0 where j > i
end

# --- main ------------------------------------------------------------------

version = begin
  XM::NArray::VERSION
rescue StandardError
  'unknown'
end

gemm_impl = if GPU
              'NArray#dot (cuBLAS)'
            elsif USE_LINALG
              'Numo::Linalg.matmul (BLAS)'
            else
              'NArray#dot (no BLAS)'
            end

puts "ruby      : #{RUBY_VERSION} (#{RUBY_PLATFORM})"
puts "backend   : #{XM} #{version}"
puts "gemm      : #{gemm_impl}"
puts "params    : T=#{T} C=#{C} NH=#{NH} HS=#{HS} LAYERS=#{LAYERS} ITER=#{ITER} " \
     "DETAIL=#{DETAIL ? 1 : 0} HEAD_COPY=#{HEAD_COPY ? 1 : 0}"
puts

w = build_weights
mask = causal_mask
x0 = XM::SFloat.new(T, C).rand * 2.0 - 1.0

# Warm up, so that creating the cuBLAS handle and the first launch of each
# kernel stay out of the measurement.
warm = XM::SFloat.new(64, 64).seq
scalar(matmul(warm, warm).sum)
sync
GC.start

# floating point operations the GEMMs do per iteration
gemm_flops = LAYERS * (2.0 * T * C * (3 * C + C + 4 * C + 4 * C) + 4.0 * T * T * C)

t0 = Process.clock_gettime(Process::CLOCK_MONOTONIC)
ITER.times do
  x = x0
  LAYERS.times { x = block(x, w, mask) }
  sync
  # Cumo leaves device memory to the Ruby GC, so put a GC.start here if this
  # runs out of memory
end
total = Process.clock_gettime(Process::CLOCK_MONOTONIC) - t0
per_iter = total / ITER

if DETAIL
  gemm_keys = %i[gemm_qkv gemm_attn gemm_proj gemm_fc]
  measured = TIMES.values.sum
  puts 'breakdown (per iteration):'
  TIMES.sort_by { |_, v| -v }.each do |key, v|
    mark = gemm_keys.include?(key) ? '*' : ' '
    puts format('  %s %-12s %8.4f s  %5.1f%%', mark, key, v / ITER, 100.0 * v / measured)
  end
  gemm_time = gemm_keys.sum { |k| TIMES[k] }
  puts format('  %s %-12s %8.4f s  %5.1f%%', '=', 'GEMM total', gemm_time / ITER,
              100.0 * gemm_time / measured)
  puts format('  %s %-12s %8.4f s  %5.1f%%', '=', 'other', (measured - gemm_time) / ITER,
              100.0 * (measured - gemm_time) / measured)
  puts format('    (* = GEMM. These sections synchronize, so they add up to more ' \
              'than the total below)')
  puts
end

puts format('total     : %.4f s / iter  (%d iter, %.4f s)', per_iter, ITER, total)
puts format('gemm      : %.2f GFLOP / iter', gemm_flops / 1e9)
puts format('throughput: %.2f GFLOP/s (the GEMM count over the total time)', gemm_flops / per_iter / 1e9)
