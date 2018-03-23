require 'rbconfig.rb'
require "erb"
require_relative '../../3rd_party/mkmf-cu/lib/mkmf-cu'

if RUBY_VERSION < "2.0.0"
  puts "Cumo::NArray requires Ruby version 2.0 or later."
  exit(1)
end

def find_narray_h
  $LOAD_PATH.each do |x|
    if File.exist? File.join(x,'numo/numo/narray.h')
      $INCFLAGS = "-I#{x}/numo " + $INCFLAGS
      break
    end
  end
end

def find_libnarray_a
  $LOAD_PATH.each do |x|
    if File.exist? File.join(x,'numo/libnarray.a')
      $LDFLAGS = "-L#{x}/numo " + $LDFLAGS
      break
    end
  end
end

def create_depend
  message "creating depend\n"
  depend_path = File.join(__dir__, "depend")
  File.open(depend_path, "w") do |depend|
    depend_erb_path = File.join(__dir__, "depend.erb")
    File.open(depend_erb_path, "r") do |depend_erb|
      erb = ERB.new(depend_erb.read)
      erb.filename = depend_erb_path
      depend.print(erb.result)
    end
  end
end

rm_f 'include/cumo/extconf.h'

if ENV['DEBUG']
  $CFLAGS="-g -O0 -Wall"
end
#$CFLAGS=" $(cflags) -O3 -m64 -msse2 -funroll-loops"
#$CFLAGS=" $(cflags) -O3"
$INCFLAGS = "-Iinclude -Inarray #{$INCFLAGS}"

$INSTALLFILES = Dir.glob(%w[include/cumo/*.h include/cumo/types/*.h include/cumo/cuda/*.h]).map{|x| [x,'$(archdir)'] }
$INSTALLFILES << ['include/cumo/extconf.h','$(archdir)']
if /cygwin|mingw/ =~ RUBY_PLATFORM
  $INSTALLFILES << ['libcumo.a', '$(archdir)']
end

srcs = %w(
cumo
narray/narray
narray/array
narray/step
narray/index
narray/ndloop
narray/data
narray/types/bit
narray/types/int8
narray/types/int16
narray/types/int32
narray/types/int64
narray/types/uint8
narray/types/uint16
narray/types/uint32
narray/types/uint64
narray/types/sfloat
narray/types/dfloat
narray/types/scomplex
narray/types/dcomplex
narray/types/robject
narray/types/bit_kernel
narray/types/int8_kernel
narray/types/int16_kernel
narray/types/int32_kernel
narray/types/int64_kernel
narray/types/uint8_kernel
narray/types/uint16_kernel
narray/types/uint32_kernel
narray/types/uint64_kernel
narray/types/sfloat_kernel
narray/types/dfloat_kernel
narray/types/scomplex_kernel
narray/types/dcomplex_kernel
narray/types/robject_kernel
narray/math
narray/SFMT
narray/struct
narray/rand
cuda/cublas
cuda/driver
cuda/memory_pool
cuda/memory_pool_impl
cuda/runtime
cuda/nvrtc
)

if RUBY_VERSION[0..3] == "2.1."
  puts "add kwargs"
  srcs << "kwargs"
end

$objs = srcs.map {|src| "#{src}.o" }

dir_config("narray")

find_narray_h
if !have_header("numo/narray.h")
  puts "
  Header numo/narray.h was not found. Give pathname as follows:
  % ruby extconf.rb --with-narray-include=narray_h_dir"
  exit(1)
end

if RUBY_PLATFORM =~ /cygwin|mingw/
  find_libnarray_a
  unless have_library("narray","nary_new")
    puts "libnarray.a not found"
    exit(1)
  end
end

if have_header("dlfcn.h")
  exit(1) unless have_library("dl")
  exit(1) unless have_func("dlopen")
elsif have_header("windows.h")
  exit(1) unless have_func("LoadLibrary")
end

if have_header("stdbool.h")
  stdbool = "stdbool.h"
else
  stdbool = nil
end

if have_header("stdint.h")
  stdint = "stdint.h"
elsif have_header("sys/types.h")
  stdint = "sys/types.h"
else
  stdint = nil
end

have_type("bool", stdbool)
unless have_type("u_int8_t", stdint)
  have_type("uint8_t",stdint)
end
unless have_type("u_int16_t", stdint)
  have_type("uint16_t",stdint)
end
have_type("int32_t", stdint)
unless have_type("u_int32_t", stdint)
  have_type("uint32_t",stdint)
end
have_type("int64_t", stdint)
unless have_type("u_int64_t", stdint)
  have_type("uint64_t", stdint)
end
have_func("exp10")

have_var("rb_cComplex")
have_func("rb_thread_call_without_gvl")

create_header('include/cumo/extconf.h')
$extconf_h = nil # nvcc does not support #include RUBY_EXTCONF_H

create_depend

HEADER_DIRS = (ENV['CPATH'] || '').split(':')
LIB_DIRS = (ENV['LIBRARY_PATH'] || '').split(':')
dir_config('cumo', HEADER_DIRS, LIB_DIRS)

have_library('cuda')
have_library('cudart')
have_library('nvrtc')
have_library('cublas')
# have_library('cusolver')
# have_library('curand')

create_makefile('cumo')
