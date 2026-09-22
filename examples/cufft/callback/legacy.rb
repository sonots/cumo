# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/legacy.py

  It runs cupy.fft.fftn with a load callback given as a plain string to
  cupy.fft.config.set_cufft_callbacks and inspects the plan cache. Cumo has
  no FFT at all (no cuFFT binding), no plan cache, and cuFFT callbacks
  additionally need a static cuFFT link step through nvcc.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
