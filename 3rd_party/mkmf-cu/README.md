# About mkmf-cu

mkmf-cu is a gem to build Ruby extensions written in C/C++ with NVIDIA CUDA.
It consists of a simple wrapper command for nvcc and a monkey patch for mkmf.

## How to use it.

Instead of require "mkmf", just
```ruby
require "mkmf-cu"
```

## How does it work?

By requiring "mkmf-cu", compiler commands defined in mkmf
will be replaced with mkmf-cu-nvcc, a command included in this gem.

When mkmf-cu-nvcc is called with arguments for gcc or clang,
it convert them to ones suitable for nvcc and execute nvcc with them.

For example,

    mkmf-cu-nvcc -I. -fno-common -pipe -Os -O2 -Wall -o culib.o -c culib.cu

will execute

    nvcc -I. -O2 -o culib.o -c culib.cu --compiler-options -fno-common --compiler-options -Wall

## Example

* https://github.com/ruby-accel/ruby-cuda-example

## Notice

When the suffix of the name of a file containing CUDA code is not .cu,
you must add the option "-x cu" to $CFLAGS or $CXXFLAGS in extconf.rb.