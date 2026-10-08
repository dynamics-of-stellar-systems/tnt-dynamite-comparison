# TNT vs. DYNAMITE comparisons

Numerical comparisons between [TNT](https://github.com/dynamics-of-stellar-systems/tnt)
and [DYNAMITE](https://github.com/dynamics-of-stellar-systems/dynamite), used to
validate that a TNT reimplementation of a DYNAMITE algorithm reproduces it
numerically.

This is **not a dependency of TNT's own test suite**. DYNAMITE (its compiled
Fortran binaries and runtime environment) is not something TNT's CI or
contributors need installed for standard development or testing. Comparisons
here are run manually, on demand, whenever a TNT feature claims numerical
parity with a DYNAMITE algorithm.

## Layout

One directory per comparison, named for what it covers, e.g.:

```
orbit-start-spaces/
  README.md          # what was compared, against which DYNAMITE commit, and the result
  run_comparison.py   # reproduces the comparison from scratch
  results/             # saved TNT and DYNAMITE output arrays
```

Each comparison's own `README.md` should record:

- which TNT commit/branch/PR it validates;
- which DYNAMITE commit it was run against (pin an exact SHA -- DYNAMITE's
  `legacy_fortran` has changed behavior across versions before);
  how to reproduce it (environment, exact commands);
- the numerical result (agreement to N significant figures, or a documented
  discrepancy).

## Adding a new comparison

1. Create a new top-level directory for it.
2. Pin the exact DYNAMITE commit SHA you compiled against in that directory's
   own README.
3. Keep the reproduction script and result arrays small enough to commit
   directly; link out (e.g. to a Claude artifact or external storage) for
   large datasets or plots rather than committing them here.
