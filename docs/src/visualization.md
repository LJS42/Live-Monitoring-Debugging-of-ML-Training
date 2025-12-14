# Visualization

This page describes the visualization components of the package.

---

## Dashboard

The dashboard displays multiple plots that update live during training.

Currently implemented plots:
- Training loss vs step
- Gradient norm vs step

---

## Makie integration

The package uses Makie.jl and observable variables to enable live updates of plots
as training progresses.
