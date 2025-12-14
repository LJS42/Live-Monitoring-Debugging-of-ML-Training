# Architecture

This page describes the internal architecture of `LMD4MLTraining.jl`.

---

## Overview

The package is organized into the following components:

- Training backend (Flux integration)
- Quantities (metrics computed during training)
- Visualization (Makie-based dashboard)
- Session management

---

## Module structure

- `cockpit/`: core abstractions (session, quantities, instruments)
- `visualization/`: dashboard and plotting code
- `backends/`: training loop integrations

This modular design allows new quantities and visual instruments to be added
without modifying the training loop logic.
