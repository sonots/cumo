#! /usr/bin/env ruby
# frozen_string_literal: true

thisdir = File.dirname(__FILE__)
libpath = File.absolute_path(File.dirname(__FILE__)) + "/../../../../lib"
$LOAD_PATH.unshift libpath

require_relative "narray_def"

$line_number = false

while true
  if ARGV[0] == "-l"
    $line_number = true
    ARGV.shift
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
  puts "usage:\n  ruby #{$0} [-l] erb_base [type_file]"
  exit 1
end

type_file, = ARGV
type_name = File.basename(type_file, ".rb")

erb_dir = ["tmpl"]
erb_dir.unshift("tmpl_bit") if (type_name == "bit")
erb_dir.map! { |d| File.join(thisdir, d) }

code = DefLib.new do
  set line_number: $line_number
  set erb_dir: erb_dir
  set erb_suffix: ".c"
  set ns_var: "mCumo"

  set file_name: $output || ""
  set type_name: type_name
  set lib_name: "cumo_" + type_name

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
