# example

Small, fast examples to see how the project works end-to-end.

## poisson_2peak

Runs the full SFRBE pipeline (PDAD -> SFR -> RBE -> evaluate) on a coarse
60x60 grid with 600 features and one RBE round (~1 minute):

```matlab
cd example
example_poisson_2peak
```

Results are saved to `results/example_poisson_2peak/` (does not overwrite the
real experiment results).