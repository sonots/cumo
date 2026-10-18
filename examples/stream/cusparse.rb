# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/stream/cusparse.py

  It builds a cupyx.scipy.sparse.csc_matrix and runs cupyx.cusparse.cscsort
  inside cupy.cuda.stream.Stream. Cumo has streams but no sparse matrix
  class and no cuSPARSE binding.
MSG
