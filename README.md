# Cumo

Cumo (pronounced "koomo") is a CUDA-aware, GPU-optimized numerical library that offers a significant performance boost over [Ruby Numo](https://github.com/ruby-numo), while (mostly) maintaining drop-in compatibility.

<img src="https://raw.githubusercontent.com/sonots/cumo-logo/master/logo_transparent.png" alt="cumo logo" title="cumo logo" width="50%">

## Requirements

* Ruby 3.0 or later
* NVIDIA GPU Compute Capability 3.5 (Kepler) or later
* CUDA 11.0 or later
* cuDNN 8.0 or later (optional, for the cuDNN features)

## Preparation

Install CUDA and set your environment variables as follows:

```bash
export CUDA_PATH="/usr/local/cuda"
export CPATH="$CUDA_PATH/include:$CPATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
export LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/lib:$LIBRARY_PATH"
```

To use cuDNN features, install cuDNN and set your environment variables as follows:

```
export CUDNN_ROOT_DIR=/path/to/cudnn
export CPATH=$CUDNN_ROOT_DIR/include:$CPATH
export LD_LIBRARY_PATH=$CUDNN_ROOT_DIR/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDNN_ROOT_DIR/lib64:$LIBRARY_PATH
```

FYI: I use [cudnnenv](https://github.com/unnonouno/cudnnenv) to install cudnn under my home directory like `export CUDNN_ROOT_DIR=/home/sonots/.cudnn/active/cuda`.

## Installation

Add the following line to your Gemfile:

```ruby
gem 'cumo'
```

And then execute:

    $ bundle

Or install it yourself as:

    $ gem install cumo

## How To Use

### Quick start

An example:

```ruby
[1] pry(main)> require "cumo/narray"
=> true
[2] pry(main)> a = Cumo::DFloat.new(3,5).seq
=> Cumo::DFloat#shape=[3,5]
[[0, 1, 2, 3, 4],
 [5, 6, 7, 8, 9],
 [10, 11, 12, 13, 14]]
[3] pry(main)> a.shape
=> [3, 5]
[4] pry(main)> a.ndim
=> 2
[5] pry(main)> a.class
=> Cumo::DFloat
[6] pry(main)> a.size
=> 15
```

### Switching from Numo to Cumo

The following find-and-replace should just work:

```
find . -type f | xargs sed -i -e 's/Numo/Cumo/g' -e 's/numo/cumo/g'
```

If you want to dynamically switch between Numo and Cumo, something like the following will work:

```ruby
if gpu
  require 'cumo/narray'
  xm = Cumo
else
  require 'numo/narray'
  xm = Numo
end

a = xm::DFloat.new(3,5).seq
```

### Incompatibility With Numo

Numo returns a Ruby numeric object wherever a result is 0-dimensional, while Cumo returns the 0-dimensional NArray itself.
Cumo differs in this way to avoid synchronization and minimize CPU ⇄ GPU data transfer.
That is not only a cost of the port; see [Keeping Scalars On The Device](#keeping-scalars-on-the-device) and [Ruby Floats In NMath Promote To Double](#ruby-floats-in-nmath-promote-to-double) for what it buys.

The methods affected are:

* `[]` and `extract`
* `count_true` and `count_false`
* reductions down to a single value: `sum`, `prod`, `mean`, `stddev`, `var`, `rms`, `min`, `max`, `ptp`, `minmax`, `median`, `mulsum`, `dot`, `inner`
* index reductions: `max_index`, `min_index`, `argmax`, `argmin`

A 0-dimensional `Cumo::Bit` is truthy even when it holds 0, because Ruby treats every object but `nil` and `false` as true.
Comparing two scalars therefore takes the wrong branch without raising anything:

```ruby
a = Cumo::SFloat[5.0]
a[0] < 1.0                  #=> Cumo::Bit#shape=[] holding 0
(a[0] < 1.0) ? :yes : :no   #=> :yes, where Numo gives :no
```

`assert_operator(a[0], :<, 1.0)` passes for the same reason, so a test suite written for Numo can stay green against Cumo while asserting nothing.
Read the value back to the host before branching on it, or run under `compatible_mode`.

Set the `CUMO_COMPATIBLE_MODE` environment variable to `ON` to force Numo NArray compatibility (for worse performance).
Running a Numo test suite that way keeps its assertions meaningful.

You may enable or disable `compatible_mode` as:

```
require 'cumo'
Cumo.enable_compatible_mode # enable
Cumo.compatible_mode_enabled? #=> true
Cumo.disable_compatible_mode # disable
Cumo.compatible_mode_enabled? #=> false
```

You can also use the following methods which behave like Numo's NArray methods. The behavior of these methods does not depend on `compatible_mode`.

* `extract_cpu`
* `aref_cpu(*idx)`
* `count_true_cpu`
* `count_false_cpu`

```ruby
a.aref_cpu(0) < 1.0   #=> false in either mode
Float(a.sum)          #=> 7.0 in either mode
```

They are methods on an NArray, so chaining one onto a result that `compatible_mode` has already turned into a Ruby object, as in `a.sum.extract_cpu`, raises `NoMethodError` while the mode is on.
`Kernel#Float` and `Kernel#Integer` read either representation, and read Numo's too, so they are what code that runs against both libraries wants.

### Keeping Scalars On The Device

The 0-dimensional return is what lets an iterative loop stay on the GPU.
Reading a scalar back to the host waits for everything queued behind it, so every read caps how far ahead the GPU is allowed to run.
What a read costs is not a fixed price either: it is however much work happens to be queued when it is taken.

`bench/cg_bench.rb` prices this with a conjugate gradient solve, 200 iterations over a 512x512 grid on an RTX 5070 Ti Laptop:

```
scalars        convergence test    us/iter    readbacks/iter
Ruby Floats    every iteration       136.7        2.02
Ruby Floats    never                 134.4        2.02
0-dim NArray   every iteration       142.6        1.02
0-dim NArray   every 20th             65.6        0.06
0-dim NArray   never                  60.3        0.02
```

Written with Ruby Floats the loop reads back twice an iteration whatever the convergence test does, since `alpha` needs `pap` and `beta` needs `rs_new` as Floats.
Thinning the test cannot get under that floor, and keeping the scalars as 0-dimensional NArrays buys nothing on its own.
The two only pay together, and together they are worth 2.1x.
The relative residual is identical in every row.

```ruby
alpha = rs_old / pap   # a 0-dimensional NArray, divided on the device
x += p_dir * alpha     # and consumed there, without crossing the bus
```

Read the value back once the loop is done, or every k iterations if it has to test something.

### Ruby Floats In NMath Promote To Double

`Cumo::NMath` picks the module it dispatches to from every argument it is given, and a Ruby `Float` counts as a `DFloat` there.
A single-precision array therefore comes back doubled whenever a plain Float rides along, even though the arithmetic operators leave it alone:

```ruby
Cumo::NMath.atan2(a, 2.0)   #=> Cumo::DFloat
Cumo::NMath.atan2(a, b)     #=> Cumo::SFloat
a + 2.0                     #=> Cumo::SFloat
```

Numo promotes the same way, and on a CPU it costs nothing: Numo's single-precision math computes in double and narrows the result anyway.
On a GeForce card, whose double-precision rate is a sixty-fourth of its single-precision one, it costs a great deal.
512x2048 elements in place on an RTX 5070 Ti Laptop:

```
                SFloat     DFloat
a * 2.0         11.8 us    12.8 us
sqrt            11.0 us    43.0 us
sin             11.5 us    97.5 us
atan            11.1 us   122.9 us
atan2           12.5 us   192.4 us
```

Only the transcendentals pay for the promotion; a double multiply runs at the speed of a single one.
The methods a Float can reach as a second argument are `atan2`, `hypot` and `ldexp`.
`ldexp` pays a different way, since scaling by a power of two is cheap in either precision: `Cumo::NMath.ldexp(a, 2.0)` takes 268.0 us against 12.6 us for `Cumo::NMath.ldexp(a, 2)`, and the difference there is the doubled arrays it has to allocate rather than the arithmetic.

Pass a 0-dimensional array instead of a Float and the call stays single precision.
That is what `[]` hands back, so a scalar taken out of an array is already in the right form:

```ruby
two = Cumo::SFloat[2.0][0]     # a 0-dimensional Cumo::SFloat
Cumo::NMath.atan2(a, two)      #=> Cumo::SFloat, 14.9 us against 219.3 us
```

Naming the module directly works too, under both libraries:

```ruby
Cumo::SFloat::Math.atan2(a, 2.0)   #=> Cumo::SFloat
```

The 0-dimensional form has no effect under Numo, where `[]` returns a Ruby Float.

### Select a GPU device ID

Set the `CUDA_VISIBLE_DEVICES=id` environment variable, or

```
require 'cumo'
Cumo::CUDA::Runtime.cudaSetDevice(id)
```

where `id` is an integer.

### Disable GPU Memory Pool

GPU memory pool is enabled by default. To disable it, set `CUMO_MEMORY_POOL=OFF`, or:

```
require 'cumo'
Cumo::CUDA::MemoryPool.disable
```

## Documentation

See https://github.com/ruby-numo/numo-narray#documentation, replacing Numo with Cumo.

## Contributions

This project is under active development. See [issues](https://github.com/sonots/cumo/issues) for future works.

## Development

Install ruby dependencies:

```
bundle install --path vendor/bundle
```

Compile:

```
bundle exec rake compile
```

Run tests:

```
bundle exec rake test
```

Generate docs:

```
bundle exec rake docs
```

## Advanced Development Tips

### ccache

[ccache](https://ccache.samba.org/) would be useful to speedup compilation time.
Install ccache and configure with:


```bash
export PATH="$HOME/opt/ccache/bin:$PATH"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/gcc"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/g++"
ln -sf "$HOME/opt/ccache/bin/ccache" "$HOME/opt/ccache/bin/nvcc"
```

### Specify nvcc --generate-code options

```
bundle exec env CUMO_NVCC_GENERATE_CODE=arch=compute_60,code=sm_60 rake compile
```

Separate the entries with a space to build for more than one architecture:

```
bundle exec env CUMO_NVCC_GENERATE_CODE="arch=compute_75,code=sm_75 arch=compute_121,code=sm_121" rake compile
```

This is useful even on development because it makes it possible to skip JIT compilation of PTX to cubin during runtime.
Without it, and without an `nvidia-smi` to read the local compute capability from, the build covers every architecture the CUDA version supports.

### Run tests with gdb

Compile with debugging enabled:

```
bundle exec DEBUG=1 rake compile
```

Run tests with gdb:

```
bundle exec gdb -x run.gdb --args ruby test/narray_test.rb
```

You may put a breakpoint by calling `cumo_debug_breakpoint()` at C source codes.

### Run tests only a specific line
`--location` option is available as:

```
bundle exec ruby test/narray_test.rb --location 121
```

### Compile and run tests only a specific type

`DTYPE` environment variable is available as:

```
bundle exec DTYPE=dfloat rake compile
```

```
bundle exec DTYPE=dfloat ruby test/narray_test.rb
```

### Run program always synchronizing CPU and GPU

```
bundle exec CUDA_LAUNCH_BLOCKING=1
```

### Show GPU synchronization warnings

Cumo shows warnings if CPU and GPU synchronization occurs if:

```
export CUMO_SHOW_WARNING=ON
```

By default, Cumo shows warnings that occurred at the same place only once.
To show all, multiple warnings, set:

```
export CUMO_SHOW_WARNING=ON
export CUMO_SHOW_WARNING_ONCE=OFF
```

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

* [LICENSE.txt](./LICENSE.txt)
* [3rd_party/LICENSE.txt](./3rd_party/LICENSE.txt)

## Related Materials

* [Fast Numerical Computing and Deep Learning in Ruby with Cumo](https://speakerdeck.com/sonots/fast-numerical-computing-and-deep-learning-in-ruby-with-cumo) - Presentation Slide at [RubyKaigi 2018](https://rubykaigi.org/2018/presentations/sonots.html#may31)
