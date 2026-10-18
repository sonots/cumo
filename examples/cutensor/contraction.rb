# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cutensor/contraction.py

  It benchmarks cupyx.cutensor.contraction (a mode-labelled tensor
  contraction) with cupyx.time.repeat, which measures with CUDA Events. Cumo
  has CUDA events but no cuTENSOR binding.
MSG
