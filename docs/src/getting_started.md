# Getting Started

This page shows how to run `LMD4MLTraining.jl` on a small MNIST example and open a live dashboard that visualizes training dynamics in real time.

---

## Requirements

- Julia
- A working Makie backend

---

## Get the code

Clone the repository and move into the project directory:

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_NAME>
```

--- 

## Start Julia
After having gotten the project open, start Julia from the project root:

```bash
julia
```

---

## Activate project and install dependencies:

```julia
pkg> activate .
pkg> instantiate
```

---

## Load the package
Exit package mode by pressing backspace and load the package:

```julia
julia> using LMD4MLTraining
```

---

## Run the MNIST live-monitoring example

```julia
julia> include("examples/mnist.jl")
```
You should now be able to see a window with two live plots:
- Training loss versus training step 
- Gradient norm versus training step