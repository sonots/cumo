# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/stream/cusparse.py

  It builds a cupyx.scipy.sparse.csc_matrix and runs cupyx.cusparse.cscsort
  inside cupy.cuda.stream.Stream. Cumo has no sparse matrix class, no
  cuSPARSE binding and no Stream class.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
