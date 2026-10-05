#! /usr/bin/env ruby
# frozen_string_literal: true

thisdir = File.dirname(__FILE__)
libpath = File.absolute_path(File.dirname(__FILE__)) + "/../../../../lib"
$LOAD_PATH.unshift libpath

require_relative "narray_def"
require_relative "kernel_parts"

$line_number = false
$kernel_part = "main"

while true
  if ARGV[0] == "-l"
    $line_number = true
    ARGV.shift
  elsif ARGV[0] == "-p"
    ARGV.shift
    $kernel_part = ARGV.shift
  elsif ARGV[0] == "-o"
    ARGV.shift
    $output = ARGV.shift
    require "fileutils"
    FileUtils.rm_f($output)
  else
    break
  end
end

if ARGV.size != 1
  puts "usage:\n  ruby #{$0} [-l] [-p part] [-o output] type_file"
  exit 1
end

type_file, = ARGV
type_name = File.basename(type_file, ".rb")

erb_dir = ["tmpl"]
erb_dir.unshift("tmpl_bit") if (type_name == "bit")
erb_dir.map! { |d| File.join(thisdir, d) }

TEMPLATE_KERNEL_PARTS = {
  "accum" => %w[reduce reduce_nan],
  "accum_index" => %w[reduce reduce_nan],
  "accum_arg" => %w[reduce reduce_nan],
  "accum_binary" => %w[reduce reduce_nan reduce_from],
  "cum" => %w[reduce],
  "layer_norm" => %w[reduce],
  "rms_norm" => %w[reduce],
  "softmax" => %w[reduce],
  "bit_count" => %w[reduce],
  "bit_reduce" => %w[reduce],
  "bit_stat" => %w[reduce],
  "cond_binary" => %w[compare],
  "cond_unary" => %w[compare],
  "rand" => %w[misc],
  "rand_norm" => %w[misc],
  "poly" => %w[misc],
  "quantize_symmetric" => %w[misc],
  "unary_ret2" => %w[misc],
}.freeze

MATH_TEMPLATES = %w[unary_s binary_s ternary_s frexp].freeze
COMMON_MATH = %w[sqrt log log2 log10 exp exp2 sin cos tan tanh erf gelu gelu_tanh silu sigmoid softplus].freeze

abort "unknown kernel part: #{$kernel_part}" unless KERNEL_PARTS.include?($kernel_part)
undeclared_math = COMMON_MATH - File.read(File.join(thisdir, "spec.rb")).scan(/^\s*math "(\w+)"/).flatten
abort "COMMON_MATH names no math function in spec.rb: #{undeclared_math.join(', ')}" unless undeclared_math.empty?

class ErbPP
  def kernel_parts
    erb_base = get(:erb_base)
    return TEMPLATE_KERNEL_PARTS[erb_base] if TEMPLATE_KERNEL_PARTS.key?(erb_base)
    return [COMMON_MATH.include?(@opts[:name]) ? "math" : "math_rare"] if MATH_TEMPLATES.include?(erb_base)
    return @children.flat_map(&:kernel_parts).uniq unless @children.empty?
    return [@opts[:type_name] == get(:class_name) ? "main" : "cast"] if %w[store_from store_bit].include?(erb_base)

    %w[main]
  end

  def kernel_children
    @children.select { |c| c.kernel_parts.include?($kernel_part) }
  end

  def kernel_part?(nan: false, from: nil)
    $kernel_part == (from ? "reduce_from" : nan ? "reduce_nan" : "reduce")
  end

  def indexer_dims(narrow = false)
    dims = (0..opt_indexer_ndim).to_a << ""
    dims += opt_indexer_narrow_dims.map { |d| "#{d}n" } if narrow
    dims
  end

  def indexer_switch(kernel, args, narrow: nil)
    launch = "<<<grid_dim, block_dim, 0, cumo_cuda_stream()>>>(#{args}); break;"
    cases = (0..opt_indexer_ndim).map do |d|
      if narrow && opt_indexer_narrow_dims.include?(d)
        "case #{d}: if (cumo_na_indexer_is_narrow(indexer, narrow_, #{narrow.size})) { #{kernel}_dim#{d}n#{launch} } #{kernel}_dim#{d}#{launch}"
      else
        "case #{d}: #{kernel}_dim#{d}#{launch}"
      end
    end
    switch = "switch (indexer->ndim) { #{cases.join(' ')} default: #{kernel}_dim#{launch} }"
    narrow ? "{ const cumo_na_iarray_t* const narrow_[] = {#{narrow.join(', ')}}; #{switch} }" : switch
  end
end

class DefModule
  def method_code
    kernel_children.map { |c| c.result }.join("\n")
  end
end

code = DefLib.new do
  set line_number: $line_number
  set erb_dir: erb_dir
  set erb_suffix: "_kernel.cu"
  set ns_var: "mCumo"

  set file_name: $output || ""
  set type_name: type_name
  set lib_name: "cumo_" + type_name

  indexer_h = File.read(File.expand_path("../../../include/cumo/indexer.h", __FILE__))
  set opt_indexer_ndim: indexer_h.match(/CUMO_NA_INDEXER_OPTIMIZED_NDIM (\d+)/)[1].to_i
  set opt_indexer_narrow_dims: indexer_h.scan(/^CUMO_NA_IARRAY_AT_NARROW\((\d+)\)/).flatten.map(&:to_i).sort

  def_class do
    extend NArrayMethod
    extend NArrayType
    eval File.read(type_file), binding, type_file
    eval File.read(File.join(thisdir, "spec.rb")), binding, "spec.rb"
  end
end.result

if $output
  open($output, "w").write(code)
else
  $stdout.write(code)
end
