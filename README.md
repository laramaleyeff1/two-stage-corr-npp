# Two-stage correlated-endpoint NPP: worked example

R/Stan tutorial for the two-stage correlated-endpoint normalized power prior, which borrows historical data on an intermediate outcome (`Y0`) to estimate subgroup treatment effects on a clinical outcome (`Y1`) in Bayesian adaptive enrichment trials.

- **Stage 1** learns how treatment effects on `Y0` translate to `Y1`, using paired data from the current trial.
- **Stage 2** borrows historical `Y0` data through a normalized power prior, pushing that information onto `Y1` via the Stage 1 link.

The script walks through one analysis of one simulated dataset.

## Requirements

```r
install.packages(c("rstan", "systemfit", "MASS"))
```

## Usage

```r
source("two-stage-npp-tutorial.R")
```

## Citation

Maleyeff L, Golchi S, Moodie EEM. *[Title]*. [Journal], [Year].
