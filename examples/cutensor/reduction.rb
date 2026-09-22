# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cutensor/reduction.py

  It benchmarks cupyx.cutensor.reduction with cupyx.time.repeat, which
  measures with CUDA Events. Cumo has no cuTENSOR binding and no Event class.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
