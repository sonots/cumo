# frozen_string_literal: true

abort(<<~MSG)
  Not ported: cupy/examples/interoperability/mpi4py_multiple_devices.py

  It exchanges CuPy arrays between two MPI ranks on two GPUs with mpi4py
  (Send, Recv and an in-place Allreduce on device buffers). Ruby has no
  maintained CUDA-aware MPI binding, and Cumo exposes no device pointer for
  one to read (no __cuda_array_interface__ equivalent).

  See MISSING_FEATURES.md for the Cumo features this needs.
MSG
