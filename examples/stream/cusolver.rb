# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/stream/cusolver.py

  It runs cupy.linalg.eigh (cuSOLVER) inside cupy.cuda.stream.Stream. Cumo
  has no cuSOLVER binding (no eigh, svd, inv or solve) and no Stream class.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
