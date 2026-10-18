# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cutensor/reduction.py

  It benchmarks cupyx.cutensor.reduction with cupyx.time.repeat, which
  measures with CUDA Events. Cumo has CUDA events but no cuTENSOR binding.
MSG
