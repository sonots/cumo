# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/jit_r2c_c2r_string.py

  It runs cupy.fft.rfft and irfft with load and store callbacks compiled
  through cupy.fft.config.set_cufft_callbacks. Cumo has no FFT at all (no
  cuFFT binding), and cuFFT callbacks additionally need a static cuFFT link
  step through nvcc.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
