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

* Add `CUMO_SHOW_WARNING` and `CUMO_SHOW_WARNING_ONCE` environment variables to suppress cumo warnings.
   * They are only for debug purpose, would be removed in future.

# 0.2.0 (2018-11-12)

All tests in red-chainer passed.

Fixes:

* Fix advanced indexing
* Fix accum\_index reduction (max\_index, and min\_index)
