# bench

`compare.rb` runs each ported example twice, once under Cumo and once as the CuPy original, and reports the times the examples themselves print.

```
CUPY_PYTHON=/path/to/python-with-cupy \
CUPY_EXAMPLES=/path/to/cupy/examples \
ruby bench/compare.rb --rounds 3 [--only cg,sgemm]
```

Each example is run once on each side before the measured rounds, so the kernel caches (`~/.cumo/kernel_cache` and `~/.cupy/kernel_cache`) are warm and neither side pays for NVRTC on the first measured run. The two sides are interleaved and the order rotates every round, so a drift in clocks or in what else the machine is doing lands on both sides rather than on one.

The numbers it produced are in `../BENCHMARK.md`.
