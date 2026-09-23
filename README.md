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

Set the `CUMO_COMPATIBLE_MODE` environment variable to `ON` to force Numo NArray compatibility (for worse performance). Every such flag takes `1`, `on`, `yes` or `true` for a yes and `0`, `off`, `no` or `false` for a no, in any case; anything else keeps the default and warns.
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

#### A Length-One Axis Does Not Break Contiguity

`contiguous?` answers true for some views Numo calls false.
An axis of length one is only ever indexed at zero, so whatever stride it carries is multiplied by zero and never moves the pointer.
Cumo leaves such an axis out of the chain it walks; Numo does not, and calls the view strided because of an axis that cannot stride.

```ruby
a = Cumo::DFloat.new(1, 4).seq
a[true, 0...2].contiguous?   #=> true, where Numo gives false
a[true, 0...2].to_a          #=> [[0.0, 1.0]], the same either way

Cumo::DFloat.new(2, 4).seq[true, 0...2].contiguous?   #=> false in both
```

The elements the view holds are the same in either library.
What changes is who is willing to read them where they lie:

* `reshape!` is accepted on these views, where Numo raises
* `dot`, `gemm`, `conv` and the rest take them as they are, instead of copying them into a contiguous array first

A three-dimensional slice behaves the same way, and so does the transpose of a single row:

```ruby
Cumo::DFloat.new(1, 1, 4).seq[true, true, 0...2].contiguous?   #=> true
Cumo::DFloat.new(1, 3).seq.transpose.contiguous?               #=> true
```

Code that runs against both libraries should not read `contiguous?` and expect the same answer.
Where it wants a contiguous array it can ask for one, since `dup` answers one in either library.

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

A read costs the wait and a copy of the block it needs into pinned host memory, a few microseconds for a scalar.
Reading managed memory from the host directly would fault its page over instead, and a small block shares a page with other live blocks that the next kernel touches, so a fresh scalar cost 0.8 ms that way on the machine above, forty times what the copy costs.
The reads that answer values, from `Float(x)` and `to_a` to `each` and `inspect`, take the copy.

### Reshape Copies, Reshape! Does Not

`reshape` answers a copy of the whole array, never a view.
Slicing answers a view, so the two look alike and are not:

```ruby
a = Cumo::SFloat.new(4, 6).seq
a[0..1, true][0, 0] = 77.0   # a view, so a changes
a.reshape(2, 12)[0, 0] = 99.0  # a copy, so a does not
```

This is what Numo does too, and numpy is where the expectation comes from: there `reshape` answers a view whenever the strides allow one.
On a GPU the difference is an allocation and a copy kernel, paid every call.
`reshape!` changes the receiver in place and costs neither.
RTX 5070 Ti Laptop, `Cumo::SFloat`, 200 calls a measurement:

```
shape        reshape     reshape!    the copy allocates
1x768        5.1 us      0.04 us          3 KB
1024x768    39.1 us      0.18 us          3 MB
4096x768    58.1 us      0.22 us         12 MB
8192x768   142.7 us      0.43 us         24 MB
```

`reshape!` is host-side bookkeeping, so it stays under a microsecond whatever the array weighs.
The `reshape` column is the copy, and it grows with the bytes.

The catch is that `reshape!` changes the array everything else is holding.
It fits a temporary the calling expression owns, and not an argument, an ivar, or anything a cache still points at:

```ruby
x = a * b            # a temporary nothing else holds
x.reshape!(t, n, h)  # free
```

Where the array is not yours to change, the copy is the price of the shape.
Reach for `reshape!` when a profile says the copies are worth removing, not by default.

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

### A Transposed Operand Goes To cuBLAS As It Is

`dot` and `gemm` hand a transposed operand to cuBLAS with its transpose flag set, rather than copying it into the layout cuBLAS reads fastest.
That saves the copy and the memory it needs.
What it costs is the kernel cuBLAS then picks, which for most shapes is slower than the one it picks for an operand already laid out its way.

RTX 5070 Ti Laptop, `Cumo::SFloat`, `q[M,K].dot(k[N,K].transpose)`, medians of nine rounds:

```
M     K     N       M*K      as it is   copied first   the copy
1     64    1500         64      5.6 us       11.4 us      2.8 us
512   64    512      32,768      8.5         11.2         2.9
1500  64    1500     96,000     33.0         29.3         3.0
512   256   512     131,072     21.8         20.7         2.8
4096  64    4096    262,144    229.1        179.3         4.2
512   768   512     393,216     54.0         39.2         3.3
256   3072  768     786,432    131.1        103.0        12.5
```

The copy weighs `N * K`, and the faster kernel it buys is worth `M * N * K`, so `M` is what decides.
A matrix-vector product, where `M` is one, is the clearest case against copying: it takes twice as long that way.
Past an `M * K` of roughly fifty thousand on this card the copy starts paying for itself, and past a few hundred thousand it is worth a fifth of the time.

Where a profile says one of these multiplications matters, hand it an operand that is already contiguous:

```ruby
kt = k.transpose.dup    # or build k transposed in the first place
q.dot(kt)
```

The table above is two-dimensional, where cuBLAS is given one matrix.
A batched multiplication takes another path through the same flag, and these numbers do not cover it.

### Fused Operations

`layer_norm`, `rms_norm` and `softmax` normalize along the last axis in one kernel each, and `quantize_symmetric` takes it to 8-bit integers in one more.
Written out of the operators they take nine launches, six, five and six, and a launch costs about two microseconds whatever it is handed, so a short row pays for the launches rather than for its bytes.

```ruby
y = x.layer_norm(gamma, beta, eps: 1e-5)   # (x - mean) / sqrt(var + eps) * gamma + beta
z = x.rms_norm(gamma, eps: 1e-5)           # x / sqrt(mean(x * x) + eps) * gamma
probs = scores.softmax                     # exp(x - max) / sum(exp(x - max))
xq, scale = x.quantize_symmetric           # scale = max(|x|) / 127, xq = round(x / scale)
```

`quantize_symmetric` answers a `Cumo::Int8` shaped like self and the scale of every row, which is self's shape without its last axis. `xq * scale[false, :new]` is what the row stood for. A row of zeros has no scale to divide by and answers zero for both, and a row holding an infinity or a NaN answers that in its scale and zeros in the row, since neither is a value 8 bits could carry. The scale comes back in the class the reduction accumulates in, which is `Cumo::SFloat` for `Cumo::HFloat` and `Cumo::BFloat` and self's own otherwise.

The rounding takes a tie away from zero, which is the rule `round` follows here and the rule Ruby's `Float#round` follows. numpy and torch take a tie to the nearest even value instead, so code ported from either answers differently wherever the quotient lands exactly halfway.

```
x / scale             -2.5  -1.5  -0.5   0.5   1.5   2.5
quantize_symmetric      -3    -2    -1     1     2     3
numpy, torch            -2    -2     0     0     2     2
```

`rint` is the other rule, at the cost of writing the quantization out: `(x / scale[false, :new]).rint.clip(-127, 127)` answers what numpy answers. A tie needs `x` to be an exact odd multiple of half the scale, so whether one ever comes up is a property of the data rather than of the arithmetic. Over a million random single-precision elements the two spellings disagreed four times, and every disagreement was a tie.

On an RTX 5070 Ti Laptop, against the same arithmetic spelled with operators, in microseconds:

```
layer_norm              SFloat            DFloat            HFloat
shape              fused  written    fused  written    fused  written
1 x 768              3.9     30.1      9.7     30.3      3.9     29.8
256 x 768            6.6     22.5     22.0     35.4      7.8     22.4
4096 x 768          32.7    177.4    300.7    610.5     28.2    142.9
1 x 1000000         19.0     59.4    125.4    175.4     22.3     68.5

softmax                 SFloat            DFloat            HFloat
shape              fused  written    fused  written    fused  written
1 x 768              4.3     19.8      7.4     26.5      6.3     34.4
256 x 768            8.8     13.9     23.4     35.1      6.8     18.5
4096 x 768          49.2    122.7    318.3    549.0     28.2    109.8
1 x 1000000         27.8     44.0    108.6    148.5     32.1     53.8

rms_norm                SFloat            DFloat            HFloat
shape              fused  written    fused  written    fused  written
1 x 768              2.9     14.2      5.8     14.1      3.1     15.1
256 x 768            4.6     16.7      6.9     20.2      5.2     15.5
4096 x 768          38.5    122.5    130.8    367.0     19.2     80.8
1 x 1000000         12.5     35.8     23.7    102.2     12.0     33.2
```

```
quantize_symmetric      SFloat            DFloat            HFloat
shape              fused  written    fused  written    fused  written
1 x 768              3.7     20.2      6.0     15.7      2.3     11.9
256 x 768            4.7     15.0     20.2     21.3      4.7     13.0
4096 x 768          24.4    105.4    289.5    333.3     23.8     80.3
1 x 1000000         18.8     40.8     97.1     81.6     15.7     37.5
```

The tables were taken in separate sessions, so read each row against the row beside it and not across the tables.

`quantize_symmetric` in `Cumo::DFloat` is worth less than the others and loses outright on a million elements, whatever shape they are in. The division it does per element is what costs: on this card one row of a million takes 44.0 us to divide in double against 10.8 in single, where the reduction over the same row takes 25.3 and 15.4. Six kernels give that division a kernel of its own to fill the device with, and one kernel leaves it behind the reduction.

All three pay off in every precision, by the most where the row is short enough that the launches were all it was doing, and by the least in double, where the reduction itself costs more than the launches ever did.
The memory clock on this card steps between 9001 and 11001 MHz under a benchmark this short, and the absolute figures move with it.
The ratios hold across the steps.

All three normalize along the last axis only.
Other axes are reachable through `transpose`.
`layer_norm` divides the variance by the row length and not by one less, which is how the layer is defined and unlike `var`; `rms_norm` divides its mean square the same way.
All three answer a new array and ignore `inplace!`, and a non-contiguous receiver is copied once rather than walked.
`gamma` and `beta` must be one-dimensional, as long as the last axis, and of the same class as the receiver.

`rms_norm` is `layer_norm` without the centring, and takes no `beta`, which is the layer Llama and the models after it normalize with.
The two agree on a row whose mean is already zero and differ on every other row, so they are not interchangeable:

```ruby
x = Cumo::SFloat[[1.0, 2.0, 3.0, 4.0]]
ones = Cumo::SFloat.ones(4)
zeros = Cumo::SFloat.zeros(4)
x.rms_norm(ones)               #=> [[0.365, 0.730, 1.095, 1.461]]
x.layer_norm(ones, zeros)      #=> [[-1.342, -0.447, 0.447, 1.342]]
```

All three accumulate in single precision, so a half row that squares past the 65504 half holds is still answered the way double answers it.
bfloat16 carries a float's exponent rather than a half's, so a bfloat16 row that squares past what a float holds overflows the accumulator too: `Cumo::BFloat[[-3e20, 3e20, 3e20, -3e20]].rms_norm(ones)` answers zeros where double answers ones.

`softmax` subtracts the row maximum before exponentiating, so a row masked with `-Float::INFINITY` answers zeros rather than `NaN`.
A row that is entirely `-Float::INFINITY` answers `NaN`, as the written-out form does.

`layer_norm` and `softmax` above, with `gelu_tanh` from the section below, are where a GPT-2 style transformer spends most of its launches.
Decoding one token of GPT-2 124M written against Numo's API takes 640 of them, and calling these three instead removes 332: 200 for the layer norms, 84 for the activations and 48 for the softmaxes.
A Llama style one spends them on `rms_norm` and `silu` instead, which is why they are here.

### Two Spellings Of gelu

`Cumo::NMath` answers both definitions of the GELU activation, under names of
their own rather than a keyword, since a unary NMath function takes none.

| method | formula | matches |
| --- | --- | --- |
| `gelu` | `0.5 * x * (1 + erf(x / sqrt(2)))` | PyTorch's default, `approximate='none'` |
| `gelu_tanh` | `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))` | PyTorch's `approximate='tanh'`, and what GPT-2 was trained with |

They differ by at most 4.1e-4, which is small enough to mistake one for the
other and far too large to swap under a set of trained weights, so pick the one
the weights were trained with.

Both follow PyTorch at the edges rather than the limit: a large negative `x`
runs out of significant digits and answers zero, and `-Float::INFINITY` answers
`NaN`.

### SiLU

`Cumo::NMath.silu` is `x / (1 + exp(-x))`, the activation Llama and the models after it use where GPT-2 uses `gelu_tanh`. It is the curve `x * sigmoid(x)` names, written as one division, and the two spellings do not answer alike: see Sigmoid below.
It is also called Swish.

```ruby
Cumo::NMath.silu(x)   # x / (1 + exp(-x)), in one kernel rather than five
```

Written out of the operators it costs five launches, and one kernel runs 1.1x to 4.3x faster on an RTX 5070 Ti Laptop, in microseconds:

```
silu                    SFloat            DFloat            HFloat
elements           fused  written    fused  written    fused  written
768                  3.6     13.4      3.6      9.5      3.4     14.3
786432               7.0     25.9     90.5    102.4      5.2     19.2
16777216           353.2   1312.9   1984.7   3258.8    199.6    851.8
```

The three rows are three regimes rather than one curve: 768 elements pay for the launches, 786432 fit in L2 and move faster than this card reads from memory, and 16777216 are what it costs from DRAM.

It follows `torch.nn.functional.silu`, answering within one unit in the last place of the true value at the nine double points measured, as torch does.
At the edges it matches torch exactly: `-Float::INFINITY` answers `NaN`, since that is what infinity times zero is, and a large negative `x` answers a signed zero.

Where that zero starts depends on the type, and not on `exp` alone.
Single and bfloat16 reach it at -89, where `exp(-x)` passes what a float holds, and double never does.
Half reaches it at -21, because the round back to half gets there first.

### Sigmoid

`Cumo::NMath.sigmoid` is `1 / (1 + exp(-x))`, the logistic curve.

```ruby
Cumo::NMath.sigmoid(x)   # in one kernel rather than four
```

Written out of the operators it costs four launches, and one kernel runs 1.2x to 4.8x faster on an RTX 5070 Ti Laptop, in microseconds:

```
sigmoid                 SFloat            DFloat            HFloat
elements           fused  written    fused  written    fused  written
768                  2.9      9.0      2.5      8.8      2.0      9.6
786432               5.6     23.0     94.6    111.3      7.3     23.6
16777216           343.5   1234.7   1978.5   2941.1    210.5    798.1
```

The kernel keeps the exponent's argument negative, which the plain quotient does not: writing `1 / (1 + exp(-x))` out asks `exp` for a value it cannot hold once `x` is negative enough, and the quotient then answers a zero where the curve is still a number the type carries.

```ruby
Cumo::NMath.sigmoid(Cumo::SFloat[-100.0]).to_a.first            #=> 3.783506e-44
(1.0 / (1.0 + Cumo::NMath.exp(Cumo::SFloat[100.0]))).to_a.first #=> 0.0
```

Where each type reaches zero, written the one way and the other:

```
             sigmoid   written out
Cumo::HFloat   -17.5         -11.5
Cumo::BFloat   -93.0         -89.0
Cumo::SFloat  -104.0         -89.0
Cumo::DFloat  -745.0        -710.0
```

A non-negative `x` takes the plain quotient unchanged, bit for bit, so only the negative half moves. It moves toward the true value more often than away, but not by much and not always: over the 360,000 single points between -88 and 0 the guarded spelling is closer at 90,423, further at 72,593 and the same at the rest, and both spellings pass one unit in the last place, the guarded one at 2,289 points and the plain one at 4,489, neither worse than three.

`silu` is `x / (1 + exp(-x))` rather than `x * sigmoid(x)`, one division rather than a division and a multiply. The second spelling rounds once more and moves a third of the answers, and it moves them in kind at the bottom: `silu` answers the signed zero torch answers below -89, where `x * sigmoid(x)` still carries a number. The two are written apart for that reason.

### Softplus

`Cumo::NMath.softplus` is `log(1 + exp(x))`, the smooth positive part, and the function a selective state space model puts its step size through.

```ruby
Cumo::NMath.softplus(x)   # log(1 + exp(x)), in one kernel rather than three
```

Written out of the operators it costs three launches, and one kernel runs 1.0x to 3.1x faster on an RTX 5070 Ti Laptop, in microseconds:

```
softplus                SFloat            DFloat            HFloat
elements           fused  written    fused  written    fused  written
768                  2.0      6.1      2.3      6.2      2.1      6.0
786432               5.9     18.3    166.6    171.8      5.7     15.0
16777216           344.2    939.5   3503.7   3938.2    206.2    591.2
```

Double barely moves at the two larger sizes, the arithmetic rather than the launches being what it pays for there.

Writing it out also gives out earlier than the kernel does, because the intermediate is an array of the receiver's type:

```ruby
x = Cumo::HFloat[22.26]
Cumo::NMath.log(1.0 + Cumo::NMath.exp(x))   #=> Infinity
Cumo::NMath.softplus(x)                     #=> 22.265625
```

Half holds `exp(x)` only to 11.09 and single to 88.7, where softplus is `x` to the last bit. The kernel takes the single-precision `exp` whatever the type, and hands back `x` where even that has no value to give, so nothing overflows at either width.

It is not `log1p(exp(x))`, which is a different number: the sum is taken before the logarithm, as the definition reads and as the implementations this follows compute. `torch.nn.functional.softplus` takes the other spelling and switches to `x` above 20, so the two differ by about an ulp where both are finite.

Taking the sum first costs the other end. Once `exp(x)` falls under the type's epsilon the sum drops it, so softplus reaches zero while the true value is still a number:

```ruby
Cumo::NMath.softplus(Cumo::SFloat[-17.0])   #=> 0.0, where the value is 4.1e-08
Cumo::NMath.softplus(Cumo::DFloat[-37.0])   #=> 0.0, where the value is 8.5e-17
```

Single and both sixteen-bit types reach that zero at -17 and double at -37, and the error grows before it: 5.9% at -16 in single and 4.3% at -36 in double. `log1p` is what to reach for where a small negative `x` has to keep its digits.

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
An input over 65504 is a different matter: it becomes an infinite element on the way in, and `var` and `stddev` answer `NaN` for a row holding one.
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
cuDNN chooses its algorithm from the ones that fit in a scratch buffer, and the half algorithms that use the tensor cores ask for a lot of it.
The default ceiling reaches them; the 8MB one Cumo used to ship does not.
N=32, C=K=64, 56x56, 3x3:

```
                                         HFloat     BFloat     SFloat
CUMO_CUDNN_MAX_WORKSPACE_SIZE=8388608    1.27 ms    1.26 ms    1.19 ms
unset (128MB)                            0.54 ms    0.60 ms    1.19 ms
CUMO_CUDNN_MAX_WORKSPACE_SIZE=268435456  0.52 ms    0.59 ms    0.59 ms
```

This shape's single-precision algorithm wants more than the default, which is the other half of the reason to look at the ceiling for a network that spends its time in `conv`.

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

### Brain Float

`Cumo::BFloat`, also reachable as `Cumo::BFloat16`, holds bfloat16: one sign bit, eight of exponent and seven of mantissa.
It is the other sixteen-bit float, and it differs from `Cumo::HFloat` in where those bits went.
A bfloat16 has the exponent of a `Cumo::SFloat` and three fewer mantissa bits than a binary16, so it reaches everything a single-precision value reaches and resolves less of it:

```ruby
Cumo::BFloat[1.0e38]   #=> 9.969209968386869e+37
Cumo::HFloat[1.0e38]   #=> Infinity

Cumo::BFloat[257.0]    #=> 256.0
Cumo::HFloat[257.0]    #=> 257.0
```

Integers are exact only to 256, against 2048 for a binary16, and the largest finite value is 3.3895314e+38 rather than 65504.
That trade is why weights are published in it: a tensor that came out of training keeps its magnitudes, and the bits it loses are ones a trained weight does not carry.

The two do not contain each other, so an expression mixing them promotes to `Cumo::SFloat`, which does:

```ruby
(Cumo::BFloat[1.0] + Cumo::HFloat[1.0]).class   #=> Cumo::SFloat
```

Everything else promotes as `Cumo::SFloat` does, so an integer array or a Ruby Float mixed in stays bfloat16 while anything wider takes over.

`layer_norm`, `rms_norm`, `softmax`, `silu`, `softplus`, `sigmoid` and both spellings of `gelu` take it, and so do the reductions, `sort`, `median`, `cumsum`, `rand` and `dot`.
The cuDNN methods take it too: `conv`, `conv_transpose`, `conv_grad_w`, `max_pool`, `avg_pool` and the three batch norm entries.

Reductions accumulate in single precision and round once at the end, exactly as they do for half, so a sum passes 256 without stopping there:

```ruby
Cumo::BFloat.new(40_000).fill(1.0).sum   #=> 39936.0
```

That 39936 is 40000 rounded to the nearest bfloat16, whose values are 256 apart up there; a bfloat16 accumulator would have answered 256.
`sum`, `mean`, `var`, `stddev`, `rms`, `mulsum`, `dot`, `gemm` and `cumsum` all widen this way, and `prod` and `cumprod` do not, for the same reasons given under Half Precision.

The wider exponent moves where the saturation described there bites.
A variance is far from 3.4e+38, so `var` does not saturate on any input a network produces, and squaring is safe to 1.8e+19 rather than to 256.
What replaces it is the mantissa: a bfloat16 resolves about three decimal digits, so a normalization computes its statistics well enough while the values it writes back carry that resolution and no more.

#### What bfloat16 is faster at

`gemm` reaches the tensor cores and lands on the same throughput as half. Square matrices on an RTX 5070 Ti Laptop, median of five runs each, one process per case:

```
              BFloat              HFloat              SFloat
1024x1024     0.044 ms  48.9 TF   0.044 ms  48.5 TF   0.144 ms  14.9 TF
2048x2048     0.298 ms  57.6 TF   0.303 ms  56.7 TF   1.404 ms  12.2 TF
4096x4096     2.839 ms  48.4 TF   2.898 ms  47.4 TF   9.540 ms  14.4 TF
```

The odd-column penalty is the same as half's and for the same reason, a row starting on a two-byte boundary.
1024x1024 times 1024x1024, with one dimension made odd at a time:

```
             all even   M odd      K odd      N odd
BFloat        48.9 TF   48.7 TF    23.5 TF    23.0 TF
HFloat        48.5 TF      -       24.0 TF    23.1 TF
```

So the choice between the two sixteen-bit types is about range and resolution, not speed.
Take bfloat16 where the magnitudes came from somewhere else and binary16 where three more mantissa bits are worth having.

#### The accelerated paths want Ampere

cuBLAS supports `CUDA_R_16BF`, which is what `gemm` and `dot` reach, from compute capability 8.0.
Storing, casting and elementwise arithmetic have no such floor, because every operation is computed in single precision and rounded back, and they build and run wherever cumo does.
A `dot` on a pre-Ampere card is the one to expect trouble from.
**This is not measured here**: the only GPU these numbers came from is a Blackwell one, and the requirement is read from cuBLAS's documentation rather than reproduced.

cuDNN reaches bfloat16 as `CUDNN_DATA_BFLOAT16`, and its own bfloat16 kernels want Ampere for the same reason cuBLAS does.
A convolution is given `CUDNN_DATA_FLOAT` to accumulate in, so it passes 256 the way a reduction does.
The batch norm parameters are `Cumo::SFloat`, the same as they are for `Cumo::HFloat`.
The workspace ceiling matters here as much as it does for half, and for the same reason.
See [Raise the cuDNN workspace ceiling](#raise-the-cudnn-workspace-ceiling).

#### An index rounds here rather than saturating

[Half Precision](#half-precision) says an index has to be built in a wider type and cast afterwards. That holds here at 256 rather than 2048, and it goes wrong more quietly:

```ruby
Cumo::BFloat[79800]   #=> 79872.0
Cumo::HFloat[79800]   #=> Infinity
```

A binary16 hands back an infinity, which the next operation carries somewhere visible. A bfloat16 hands back a plausible integer, and whatever reads it goes on. Where the index is an angle, the answer comes back with the wrong sign:

```ruby
Cumo::NMath.cos(Cumo::SFloat[79800])   #=> -0.9190999865531921
Cumo::NMath.cos(Cumo::SFloat[79872])   #=> 0.989012598991394
```

Not every integer past 256 is lost, which is what makes this one hard to catch by sampling: `Cumo::BFloat[1000]` is exact, since 1000 is a multiple of 8.
What ends at 256 is that consecutive integers stay distinct.

### Launching Your Own Kernel

A kernel written in CUDA C can be compiled with NVRTC and launched on the arrays.
It runs on the stream Cumo's own kernels use, so it sees the results of the operations issued before it and the operations after it see its.

```ruby
source = <<~CUDA
  extern "C" __global__ void axpy(float* y, const float* x, float a, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
      y[i] = a * x[i] + y[i];
    }
  }
CUDA
mod = Cumo::CUDA::Compiler.new.compile_with_cache(source)
axpy = mod.get_function("axpy")

x = Cumo::SFloat.new(1000).seq
y = Cumo::SFloat.ones(1000)
axpy.launch([y, x, [2.5].pack("f"), [1000].pack("l")], grid: 4, block: 256)
```

`compile_with_cache` compiles once and keeps the cubin on disk, and within a process the same source and options answer the same module, so it can be called wherever the kernel is launched.
That module is shared by everything in the process that compiles the source, so unload it only when nothing else uses it; a module nobody holds any more is unloaded when Ruby collects it.
`Cumo::CUDA::Compiler.clear_modules` makes the next call read the disk cache or compile again.
An NArray argument hands over its device pointer, so it has to be contiguous.
An Integer is passed as a `long long` and a Float as a `double`.
Anything narrower, and a struct passed by value, goes as the packed bytes of a String: `[n].pack("l")` is an `int` and `[x].pack("f")` a `float`.
`grid` and `block` take one to three sizes each, and `shared_mem:` is the dynamic shared memory in bytes.
The kernel is asked for by the name in the source, so it is declared inside `extern "C"`, or asked for by its mangled name.
On CUDA 12.4 or later the count and the size of the arguments are checked against the kernel before the launch, so an Integer handed to an `int` is refused rather than read wrong.
A kernel may write any NArray it is handed, so a frozen one is refused, and one that has not been allocated yet is allocated on the way and holds whatever was there.

### Streams And Events

Every kernel and copy Cumo issues goes to the current stream of the thread, which is the null stream until a `Cumo::CUDA::Stream` is used.
`Stream#with` runs a block on a stream of its own, waits for everything the block queued, and puts the previous stream back, so what the block produced is complete when it returns.
The block's work is ordered after what the previous stream had queued, so an input still being computed when the block starts is read complete.

```ruby
s = Cumo::CUDA::Stream.new(non_blocking: true)
c = s.with { a.gemm(b) }   # queued on s, and finished when with returns
```

`Stream#use` makes a stream current without a block, `Stream.current` answers the current one and `Stream.null` the null stream.
`Stream#record` records an `Event` after everything queued so far, and `Stream#wait_event` makes what is queued after it wait for one, which is how two streams are ordered against each other.
`Event#synchronize` waits for an event on the host, and `Cumo::CUDA.get_elapsed_time(start, stop)` answers the milliseconds between two recorded events, which is how a kernel is timed without a device-wide wait.

```ruby
start = Cumo::CUDA::Event.new.record
c = a.gemm(b)
stop = Cumo::CUDA::Event.new.record
stop.synchronize
Cumo::CUDA.get_elapsed_time(start, stop)   # => milliseconds
```

Under a stream of the caller's, a host read such as `to_a` or `each` waits for the whole device rather than for that stream, since what it reads may have been written on another one.
The current stream is per thread, and fibers of one thread share it.
A `Function#launch` takes `stream:` to launch on a stream other than the current one.

### Pinned Host Memory

A copy between the host and the device is asynchronous only when the host side is page-locked.
`Cumo::CUDA::PinnedMemory` allocates such a buffer, and `NArray#set` and `NArray#get` copy through it on the current stream, or on `stream:`.

```ruby
pinned = Cumo::CUDA::PinnedMemory.new(a.byte_size)
s = Cumo::CUDA::Stream.new(non_blocking: true)
s.with { a.get(pinned) }                         # queued on s, complete when with returns
b = Cumo::SFloat.from_binary(pinned.read, a.shape)

src = Cumo::CUDA::PinnedMemory.from_binary(bytes)
c.set(src, stream: s)                             # the host bytes reach c once s gets there
```

`set` with a String and `get` with nothing to copy into are the synchronous `store_binary` and `to_binary`.
A pinned buffer is read and written as bytes with `read` and `write`, and the array has to be contiguous and of the buffer's size.
A copy in flight keeps its array alive and is waited for by `read`, `write`, `free` and the next copy, so the bytes read are the copy's.
Allocating a pinned buffer synchronizes the device, so a buffer is allocated once and reused rather than made for every transfer.

### Structs And Template Kernels

An array of structs is an array whose trailing axes are the struct's fields, so a kernel that takes `const double3*` is handed a `Cumo::DFloat` of shape `[n, 3]`, and one that takes `const Matrix<float>*` an `SFloat` of shape `[n, 4, 4]`.
`SComplex` and `DComplex` are `float2` and `double2` to a kernel; `Bit`, which packs its elements, and `RObject` cannot be handed to one.
A struct passed by value is the packed bytes of its fields, and where the struct has padding, `pack`'s `@` places each field at the offset the device reports.

```ruby
rhs = [x, y, z].pack("d3")                           # a double3 by value
sum_kernel.launch([lhs, rhs, out], grid: 1, block: n)
```

A template kernel has no `extern "C"` name.
`compile_with_cache` takes `name_expressions:`, and the module then answers `get_function` for each expression by the mangled name NVRTC reports.

```ruby
mod = Cumo::CUDA::Compiler.new.compile_with_cache(source, name_expressions: ["kernel<float>", "kernel<double>"])
mod.get_function("kernel<float>").launch([a, b, c.to_a.flatten.pack("f*"), out], grid: 1, block: n)
```

### Writing An Elementwise Kernel

`Cumo::CUDA::ElementwiseKernel` takes one piece of CUDA C and applies it to every element, the way CuPy's `ElementwiseKernel` does.
The kernel is compiled once for each set of dtypes, once more when a different argument is a number or is broadcast, and kept for the next call.

```ruby
squared_diff = Cumo::CUDA::ElementwiseKernel.new(
  "T x, T y", "T z", "z = (x - y) * (x - y)", "squared_diff")

x = Cumo::SFloat.new(2, 5).seq
y = Cumo::SFloat.new(5).seq
squared_diff.call(x, y)   # => the (2, 5) array of squared differences
squared_diff.call(x, 5)   # => the same against a scalar
```

A type is one of `float64`, `float32`, `int64`, `int32`, `int16`, `int8`, `uint64`, `uint32`, `uint16` and `uint8`, or a single letter that stands for whichever dtype the argument has.
Outputs decide a letter before inputs do, and a letter that only a Ruby number reaches becomes `int64` or `float64`.
Array arguments are broadcast against each other, and an output may be given after the inputs, or is allocated.
An input is read through its own strides, so a transposed, reversed or stepped view goes to the kernel as it is.
A view built on an index array is copied first, and so is an input that shares memory with an output other than element for element, since the kernel would read what another thread has already written; a raw argument has to be contiguous and is copied when it is not.
An argument marked `raw T y` is handed over as a pointer for the operation to index itself, with `i` the element index and `_ind.size()` the element count, and when every argument is raw or a number, `size:` says how many elements there are.
Inputs are `const`, so an operation that writes one does not compile, and a number handed to an integer type has to be an Integer that fits.
`preamble:` is placed before the kernel, after the typedefs of the letters, so a device function can be written in terms of `T`.
`Cumo::Bit`, `Cumo::HFloat`, `Cumo::BFloat`, `Cumo::SComplex`, `Cumo::DComplex` and `Cumo::RObject` cannot be handed to one of these kernels yet.

### Writing A Reduction Kernel

`Cumo::CUDA::ReductionKernel` reduces along axes with three expressions, the way CuPy's `ReductionKernel` does: a map applied to every element, a reduce between two mapped values `a` and `b`, and a post map that writes the reduced value `a` to the output.
The identity starts every reduction.

```ruby
l2norm = Cumo::CUDA::ReductionKernel.new(
  "T x", "T y", "x * x", "a + b", "y = sqrt(a)", "0", "l2norm")

x = Cumo::SFloat.new(2, 5).seq
l2norm.call(x, axis: 1)   # => [5.477, 15.969]
l2norm.call(x)            # => the 0-dimensional norm of everything
```

The parameters, the types and the broadcasting follow `ElementwiseKernel`, and `axis:` and `keepdims:` follow `sum`.
`reduce_type:` names the type the values are accumulated in, as a C type, a type name or one of the letters, and is the output's type unless said otherwise.
The post map may be several statements, so a kernel can write several outputs.
A `raw` parameter is not taken, since a reduction indexes every argument itself.
A long axis reduced to a few outputs is split across blocks and folded in a second pass, so it runs as fast as `sum`.

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

### Reading The Memory Numbers

cumo allocates with `cudaMallocManaged`, so a page can sit on the host, and `nvidia-smi` counts only what the card is holding at that moment.
It reports less than cumo has taken, and the gap is not a fixed one.
A library that allocates with `cudaMalloc` reports everything it took, so the two numbers do not belong side by side in a comparison.

From inside, the pool answers two different questions:

```ruby
Cumo::CUDA::MemoryPool.total_bytes   # what the pool has taken from the card
Cumo::CUDA::MemoryPool.used_bytes    # what it has handed out and not taken back
```

`used_bytes` still counts blocks whose last reference is gone but which Ruby's garbage collector has not reached, so read it after `GC.start` when you mean the arrays that are alive:

```ruby
a = Cumo::SFloat.new(1024, 1024).seq
t = a + 1
t = nil
Cumo::CUDA::MemoryPool.used_bytes    #=> 8388608
GC.start
Cumo::CUDA::MemoryPool.used_bytes    #=> 4194304
```

`total_bytes` does not move with the collector and counts the free blocks the pool is keeping, so it answers how much this took rather than how much is alive.

## Examples

`examples/` holds ports of the examples in the CuPy repository, one Ruby file per original, with a `backend.rb` that runs the CPU side on Numo and the GPU side on Cumo under `GPU=1`.
`examples/README.md` lists what is ported and how to run them.

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

cuDNN picks a convolution algorithm by benchmarking the ones that fit in a scratch buffer, and the ceiling on that buffer is 128MB.
The search reserves the whole ceiling whatever the convolution's size and hands it back to the pool afterwards, so the ceiling costs a peak rather than a residency.

Some shapes want more than the default. The twenty convolutions of a ResNet-18 forward pass at batch 16, and the single-precision case from [Half Precision](#half-precision):

```
                                         ResNet-18    N=32, C=K=64, 56x56, 3x3
CUMO_CUDNN_MAX_WORKSPACE_SIZE=8388608      9.80 ms                     1.19 ms
unset (128MB)                              5.23 ms                     1.19 ms
CUMO_CUDNN_MAX_WORKSPACE_SIZE=268435456    5.24 ms                     0.59 ms
```

To raise it:

```
export CUMO_CUDNN_MAX_WORKSPACE_SIZE=268435456
```

The value is in bytes and only bounds the search; each convolution reserves what its chosen algorithm actually needs.
`Cumo::CUDA::CUDNN.max_workspace_size` reads back the value in force.

### Single precision stays off the tensor cores

cuDNN reads its default math mode as "tensor cores are allowed", so a single precision convolution moves onto them as soon as the algorithm search reaches an algorithm that has them.
The operands are rounded to a 10 bit significand on the way, and nothing in the call said to do that.
Cumo asks cuDNN to keep single precision off the tensor cores, so raising the workspace ceiling buys speed at single precision accuracy.
A different algorithm rounds differently, so the answer still moves; what it does not do is drop two digits.
The half types name the tensor cores themselves and are not affected.

To trade the accuracy for the speed:

```
export CUMO_ALLOW_TF32=1
```

Measured over the convolutions of a ResNet-18 forward pass at batch 16, with the ceiling raised to 256MB, each layer against a double precision reference:

```
                              pass      worst layer
tensor cores off (default)   5.24 ms      1.4e-05
tensor cores on              4.31 ms      2.4e-04
```

The same flag puts `SFloat` and `SComplex` `gemm` on the tensor cores as TF32, and `dot` where it goes through `gemm`.
Off, the answer is the one cuBLAS gives at single precision, bit for bit.
On, a `[4096, 4096]` `SFloat` `gemm` goes from 17.8 to 28.4 TFLOP/s on an RTX 5070 Ti Laptop, 1.60 times over six interleaved rounds.
Its answer is then about 3e-04 from a double precision reference, where it was 4e-07.
The double types are not affected either way.

What the flag buys depends on where the time goes.
It pays where matrix products and convolutions take the time, as in training or in a batch through a convolutional network.
It buys nothing where reading the operands takes the time, as in decoding one token at a time, where each step reads every weight once and the GEMM waits on memory rather than on arithmetic.
Over a training run the rounding adds up, so a loss that meets a tolerance at single precision can miss it on some steps; compare a few steps with the flag off before relying on it.

`Cumo.allow_tf32?` reads back the value in force.

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/sonots/cumo.

## License

* [LICENSE.txt](./LICENSE.txt)
* [3rd_party/LICENSE.txt](./3rd_party/LICENSE.txt)

## Related Materials

* [Fast Numerical Computing and Deep Learning in Ruby with Cumo](https://speakerdeck.com/sonots/fast-numerical-computing-and-deep-learning-in-ruby-with-cumo) - Presentation Slide at [RubyKaigi 2018](https://rubykaigi.org/2018/presentations/sonots.html#may31)
