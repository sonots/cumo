# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/stream/cufft.py

  It runs cupy.fft.fft inside cupy.cuda.stream.Stream. Cumo has no FFT (no
  cuFFT binding) and no Stream class.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
