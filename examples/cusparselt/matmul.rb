# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/cusparselt/matmul.py

  It drives cuSPARSELt directly through cupy_backends.cuda.libs.cusparselt:
  handle and descriptor init, 2:4 structured pruning, compression, algorithm
  search and matmul on float16 matrices. Cumo has HFloat arrays but no
  cuSPARSELt (or cuSPARSE) binding and no way to hand a raw device pointer
  to one.

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
