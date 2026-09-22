# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/jit_string.py

  It runs cupy.fft.fftn with a load callback compiled through
  cupy.fft.config.set_cufft_callbacks and inspects the plan cache. Cumo has
  no FFT at all (no cuFFT binding), no plan cache, and cuFFT callbacks
  additionally need a static cuFFT link step through nvcc.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
