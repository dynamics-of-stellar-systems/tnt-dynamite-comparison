# Orbit start spaces

Compares TNT's three orbit-sampler initial-condition generators against
DYNAMITE's compiled `orbitstart` binary:

- **Box orbits** -- TNT's `StationaryGridOrbitSampler` vs. DYNAMITE's
  `beginbox.dat`.
- **Tube orbits** -- TNT's `XZGridFromBoundaryOrbitSampler` (the
  boundary-searched, DYNAMITE-matching sampler) vs. DYNAMITE's `begin.dat`.

Across four NGC6278-based viewing geometries (oblate, mild/strong triaxial,
near-prolate), six energy shells each (`r` from `0.01` to `100 kpc`).

## What's validated, and what to trust

- **TNT commit**: validates `dynamics-of-stellar-systems/tnt` PR #80
  (`box_tube_start_spaces`), after all six audit findings in
  `aidocs/pr-80-orbit-start-spaces-audit.md` were fixed.
- **DYNAMITE commit**: `0bd10a9586936bf6b9bcfdcd3679477d812a01d0`.
- **Result**: box orbits agree to 5-6 significant figures (median relative
  error ~3e-7, max ~5e-6); tube orbits agree to 4-5 significant figures in
  the typical (median) case (~5e-5 to 8e-5), with one consistent worst case
  around 3% at the innermost energy shell's near-origin point -- a small
  absolute difference becoming a large relative one right where both codes'
  own `r_floor`-style numerical safeguards are most active, not a sign of a
  shell-by-shell problem (see the notebook's error-summary plot and
  per-shell breakdown).

## Layout

- `input/parameters_pot.in` and `scenario_*.yaml` is DYNAMITE's own fully
  resolved per-geometry input (the exact values `orbitstart` consumed) and
  the higher-level DYNAMITE config each was generated from, kept for
  provenance.
- `mge_lum.ecsv` is the shared NGC6278 MGE (identical across all four
  geometries; only the viewing angles change) -- note this is the same
  fixture already used by TNT's own `tests/integration_tests/`.
- `dynamite_output/<geometry>/begin[box].dat` is DYNAMITE's own real
  `orbitstart` output, committed directly rather than regenerated.
- `run_comparison.py` builds the equivalent TNT potential and orbit
  samplers from `input/`, generates TNT's own initial conditions, and
  compares them against `dynamite_output/`, saving arrays and a summary to
  `results/`.
- `results/` is `run_comparison.py`'s saved output: per-geometry `.npy`
  arrays and `summary.json`.
- `orbit-start-spaces-comparison.ipynb` visualises `results/` -- `(x, z)`
  starting-position grids for both populations, a `v_y` agreement plot for
  the tube population, and the relative-error summary.

## Reproducing

```
cd orbit-start-spaces
python run_comparison.py   # needs a TNT development environment
jupyter nbconvert --to notebook --execute --inplace orbit-start-spaces-comparison.ipynb
```

The notebook only reads `results/`, so it doesn't need a TNT environment --
just `numpy`, `matplotlib`, and `jupyter`.
