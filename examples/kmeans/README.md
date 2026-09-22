# kmeans example

This example contains implementation of K-means clustering.


### How to demo
The demo contains a script that partitions data into groups using K-means clustering.
The demo can be run by the following command.

```
ruby kmeans.rb [--gpu-id GPU_ID] [--n-clusters N_CLUSTERS] [--num NUM]
               [--max-iter MAX_ITER] [--use-custom-kernel]
               [--output-image OUTPUT_IMAGE]
```

Set `GPU=1` to run the GPU side; without it only the CPU side runs. The CPU side runs on Numo and the GPU side on Cumo, from the same `fit_xp` method.

`--use-custom-kernel` is accepted but stops with a message: the CuPy original builds it from `cupy.ElementwiseKernel` and `cupy.ReductionKernel`, which Cumo does not have. `--output-image` is accepted but only prints a notice, since the drawing needs matplotlib. See `../MISSING_FEATURES.md`.
