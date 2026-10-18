# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cufft/callback/jit_r2c_c2r_string.py

  It runs cupy.fft.irfft with a load callback compiled through
  cupy.fft.config.set_cufft_callbacks. Cumo::FFT has rfft and irfft, but no
  way to attach a cuFFT callback to a plan.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
