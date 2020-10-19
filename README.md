# Source code for the experiments of the "Gaussian Particle Flow" paper submitted for AISTATS 2021.

## Installation

- Make sure to install `julia` version `1.5.2` : https://julialang.org/downloads/

- Download this repository (use the zip option)
- Go in the folder, open a terminal and run `julia` followed by the following commands (note that `]` allows you to go in package mode)

```julia
] activate .
] instantiate
] dev AdvancedVI
```

- The `AdvancedVI` package is a fork of the original `Turing.jl` package.

## Running the scripts

Once everything has been installed you can run the scripts by either running `julia` from the terminal and run the scripts via `include("scripts/run_gaussians.jl")` for example. The scripts are going to run over the different given parameters and save the resutls in `data/results` . You can then view the results by calling `include("analysis/gaussian/process_gaussians.jl")` for example.

## Datasets

The used datasets will automatically be downloaded from their source, feel free to use anything!