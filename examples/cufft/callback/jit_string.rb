# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/jit_string.py

  It runs cupy.fft.fftn with a load callback compiled through
  cupy.fft.config.set_cufft_callbacks and inspects the plan cache. Cumo::FFT
  has fftn and a plan cache, but no way to attach a cuFFT callback to a plan.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
