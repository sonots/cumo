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

### Half Precision

`Cumo::HFloat`, also reachable as `Cumo::Float16`, holds IEEE binary16: one sign bit, five of exponent and ten of mantissa.
It exists for the two things half is good at, moving half as many bytes and reaching the tensor cores, and it promotes exactly as `Cumo::SFloat` does, so an integer array or a Ruby Float mixed into an expression stays half while anything wider takes over.

What it cannot hold is the thing to plan around.
Integers are exact only to 2048, and the largest finite value is 65504:

```ruby
Cumo::HFloat[2049.0]        #=> 2048.0
Cumo::HFloat[50257.0]       #=> 50272.0
Cumo::HFloat[100000]        #=> Infinity
```

There is no exception on the way past the top; the value saturates, as it does in every other float type.
Anything that carries an index rather than a measurement has to be built in a wider type and cast afterwards.

Reductions do not inherit that limit, because they accumulate in single precision and round once at the end.
A sum of forty thousand ones is forty thousand, not the 2048 a half accumulator would stop at:

```ruby
Cumo::HFloat.new(40_000).fill(1.0).sum   #=> 40000.0
```

`sum`, `mean`, `var`, `stddev`, `rms`, `mulsum`, `dot`, `gemm` and `cumsum` all widen this way.
`prod` and `cumprod` do not: a product leaves half's range long before it loses precision, so widening the accumulator would only hide the overflow.

The widening protects the accumulation, not the answer, which is still stored as half.
A variance above 65504 therefore saturates even though nothing overflowed while it was being computed, and the standard deviation of the same array is fine because the square root brings it back into range:

```ruby
a = Cumo::HFloat[2000.0, -2000.0, 1000.0, -1000.0]
a.var      #=> Infinity
a.stddev   #=> 1826.0
```

That distinction is what makes half usable in a transformer's layer normalization, and getting it wrong is the first thing to go wrong there.
Squaring overflows at 256, since 256 squared is already past the top, and a residual stream with one outlier feature reaches thousands.
Writing the normalization out as `((x - mean) ** 2).mean` builds those squares as a half array and the answer is `Infinity`, while `x.var(axis: 1)` keeps them in its single-precision accumulator and answers 10208.0 against a single-precision 10209.
`var` saturates only if the variance itself is out of range, which is a far higher bar than any one deviation being over 256, but it is still a bar.
The quantity to check against 65504 is the largest variance a layer produces, not the largest activation in it.
A single outlier of size d among n values contributes only d squared over n to the variance, so a row 768 wide divides it by 768: an activation of 3000 squares to nine million but raises the variance of its row by about twelve thousand.
An activation that looks safe therefore says nothing about whether `var` overflows, in either direction.

A `dot` whose answer does not fit still saturates, since the result is stored back as half.
The accumulator itself is single precision, so scaling it down on the way out recovers the value:

```ruby
a = Cumo::HFloat.ones(1, 1024)
b = Cumo::HFloat.new(1024, 1).fill(100.0)
a.dot(b)                   #=> Infinity
a.gemm(b, alpha: 0.001)    #=> 102.375, the true 102400 scaled down
```

#### What half is faster at

`gemm` reaches the tensor cores. Square matrices on an RTX 5070 Ti Laptop, median of three runs each:

```
              HFloat              SFloat              DFloat
1024x1024     0.043 ms  49.7 TF   0.161 ms  13.3 TF   5.370 ms  0.40 TF
2048x2048     0.362 ms  47.4 TF   1.360 ms  12.6 TF   43.46 ms  0.40 TF
4096x4096     3.335 ms  41.2 TF   10.40 ms  13.2 TF   328.1 ms  0.42 TF
```

An odd number of columns costs half far more than it costs the others, because a row then starts on a two-byte boundary and the vectorized path is gone.
It is the column count of either operand that matters, not the row count.
1024x1024 times 1024x1024, with one dimension made odd at a time:

```
             all even   M odd      K odd      N odd
HFloat        51.8 TF   49.2 TF    23.3 TF    23.3 TF
SFloat        11.9 TF      -       11.6 TF    10.8 TF
```

The run-to-run spread on these is a few per cent and reaches fifteen at the top end, so the M column says the row count does not matter rather than that it costs 5 per cent.

The penalty is on the arithmetic, so it does not reach a matrix-vector product, which is bound by how fast the matrix can be read whatever its shape.
A 1x768 by 768x50257 gemv takes 0.211 ms with that odd 50257 and 0.205 ms with 50256, a difference inside the noise; the same 768x50257 matrix against 256 rows takes 0.833 ms and 0.481 ms, which is not.
Pad the inner dimensions of a real matrix product; leave a gemv alone.

`conv` is a different story, and worth reading before reaching for half in a network.
cuDNN chooses its algorithm from the ones that fit in a scratch buffer, and the half algorithms that use the tensor cores ask for more than the default 8MB ceiling allows.
Left at the default, a half convolution is no faster than a single-precision one.
N=32, C=K=64, 56x56, 3x3:

```
                              HFloat     SFloat
CUMO_CUDNN_MAX_WORKSPACE_SIZE unset      1.13 ms    1.03 ms
CUMO_CUDNN_MAX_WORKSPACE_SIZE=268435456  0.56 ms    0.68 ms
```

Tensor cores also want the channel counts to be multiples of eight, which the first layer of a network never satisfies.
That layer is still faster in half, but for the other reason:

```
C=3, K=64, 56x56, 3x3     0.085 ms   0.170 ms
C=K=64, 56x56, 1x1        0.037 ms   0.120 ms
```

Neither of those reaches a tensor core; they move half the bytes.

#### Batch normalization takes single-precision parameters

cuDNN derives the descriptor for the batch norm parameters from `x`, and widens it to float when `x` is half.
`gamma`, `beta`, `running_mean`, `running_var`, `mean` and `inv_std` are therefore `Cumo::SFloat` where `x` is `Cumo::HFloat`, and `batch_norm_backward` answers `gx` in half with `ggamma` and `gbeta` in single:

```ruby
x = Cumo::HFloat.new(2, 4, 3, 3).seq
gamma = Cumo::SFloat.ones(4)
beta = Cumo::SFloat.zeros(4)
x.batch_norm(gamma, beta, axis: [0, 2, 3])   #=> Cumo::HFloat
```

Passing half parameters raises `TypeError: gamma must be Cumo::SFloat, not Cumo::HFloat`.
Keeping the running statistics in single precision is what the arithmetic wants in any case: a momentum update is a long chain of small corrections, and eleven bits of mantissa lose them.

Gradients are the other place half runs out of room.
Values below `Cumo::HFloat::MIN` of 6.1e-05 fall into the subnormals and then to zero, which is what loss scaling in a training loop exists to prevent.

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

### Raise the cuDNN workspace ceiling

cuDNN picks a convolution algorithm by benchmarking the ones that fit in a scratch buffer, and the ceiling on that buffer is 8MB.
The fastest half precision algorithms, the ones that reach the tensor cores, ask for more than that and are left out of the search.
To let them in:

```
export CUMO_CUDNN_MAX_WORKSPACE_SIZE=67108864
```

The value is in bytes and only bounds the search; each convolution reserves what its chosen algorithm actually needs.
`Cumo::CUDA::CUDNN.max_workspace_size` reads back the value in force.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

* [LICENSE.txt](./LICENSE.txt)
* [3rd_party/LICENSE.txt](./3rd_party/LICENSE.txt)

## Related Materials

* [Fast Numerical Computing and Deep Learning in Ruby with Cumo](https://speakerdeck.com/sonots/fast-numerical-computing-and-deep-learning-in-ruby-with-cumo) - Presentation Slide at [RubyKaigi 2018](https://rubykaigi.org/2018/presentations/sonots.html#may31)
