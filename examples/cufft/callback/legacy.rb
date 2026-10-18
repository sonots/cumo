# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/legacy.py

  It runs cupy.fft.fftn with a load callback given as a plain string to
  cupy.fft.config.set_cufft_callbacks and inspects the plan cache. Cumo::FFT
  has no way to attach a cuFFT callback to a plan, and these legacy callbacks
  work only with the static cuFFT library, while Cumo loads the shared one.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
