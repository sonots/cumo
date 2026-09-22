# GMM example

This example contains implementation of Gaussian Mixture Model (GMM).


### How to demo
The demo contains a script that partitions data into groups using Gaussian Mixture Model.
The demo can be run by the following command.

```
ruby gmm.rb [--gpu-id GPU_ID] [--num NUM] [--dim DIM]
            [--max-iter MAX_ITER] [--tol TOL] [--output-image OUTPUT]
```

Set `GPU=1` to run the GPU side; without it only the CPU side runs. The CPU side runs on Numo (with numo-linalg for BLAS `dot`) and the GPU side on Cumo, from the same methods.

`--output-image` is accepted but only prints a notice, since the drawing needs matplotlib and scipy.stats. See `../MISSING_FEATURES.md`.
