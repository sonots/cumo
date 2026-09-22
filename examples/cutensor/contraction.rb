# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cutensor/contraction.py

  It benchmarks cupyx.cutensor.contraction (a mode-labelled tensor
  contraction) with cupyx.time.repeat, which measures with CUDA Events. Cumo
  has no cuTENSOR binding and no Event class.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
