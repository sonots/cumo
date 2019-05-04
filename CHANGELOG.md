# 0.3.4 (2019-05-04)

Enhancements:

* Support cuDNN fixed\_batch\_norm (cudnnBatchNormalizationForwardInference)

# 0.3.3 (2019-05-02)

Fixes:

* Fix each and `each_with_index` to synchronize on each element

# 0.3.2 (2019-05-02)

Fixes:

* Fix max and max\_index for sfloat and dfloat

# 0.3.1 (2019-04-16)

Fixes:

* Fix batch\_norm\_backward
* Fix scalar.dot(scalar)
* Fix clip

# 0.3.0 (2019-04-10)

Enhancements:

* Support cuDNN
  * conv (cudnnConvolution)
  * conv\_transpose (cudnnConvolutionBackwardData)
  * conv\_grad\_w (cudnnConvolutionBackwardFilter)
  * batch\_norm (cudnnBatchNormalizationForwardTraining)
  * batch\_norm\_backward (cudnnBatchNormalizationBackward)
  * avg\_pool and max\_pool (cudnnPoolingForward)
  * avg\_pool\_backward and max\_pool\_backward (cudnnPoolingBackward)

# 0.2.5 (2019-03-04)

Enhancements:

* Support arithmetic sequence, which is available in ruby >= 2.6.0 (thanks to naitoh)

# 0.2.4 (2018-11-21)

Changes:

* Turn off `CUMO_SHOW_WARNING` as default

# 0.2.3 (2018-11-17)

Enhancements:

* Add some missing `synchronize` workarounds

# 0.2.2 (2018-11-13)

Enhancements:

* CUDA kernelize na\_index\_aref\_naview
* CUDA kernelize na\_index\_aref\_nadata
* CUDA kernelize diagonal
* CUDA kernelize copy

# 0.2.1 (2018-11-12)

Enhancements:

* Add `CUMO_SHOW_WARNING` and `CUMO_SHOW_WARNING_ONCE` environment variables to suppress cumo warnings (They are only for debug purpose, would be removed in future).

# 0.2.0 (2018-11-12)

All tests in red-chainer passed.

Fixes:

* Fix advanced indexing
* Fix accum\_index reduction (max\_index, and min\_index)
