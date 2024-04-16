# MLE-Fluxes


## Context and Motivation

Purpose of this experiment is to compute the vertical buoyancy fluxes induced by different submesoscale Mixed Layer Eddies (MLE) parameterisation on a 10 years global eORCA025 config.
Internal and external computed fluxes are written in an output file with the NEMO output system (XIOS).

#### Variations
- **STD** : External standard computation, as described in [Calvert et al. (2020)](https://doi.org/10.1016/j.ocemod.2020.101678) with retroaction on the solution
- **CNN** : Fluxes computed from pre-trained [Bodner, Balwada and Zanna (2024)]() CNN `...WORK IN PROGRESS...`

## Experiments Requirements


### Compilation

- NEMO version : [v4.2.1](https://forge.nemo-ocean.eu/nemo/nemo/-/releases/4.2.1) patched with [morays](https://github.com/morays-community/morays-doc/tree/main/nemo_src) and local `CONFIG/src` sources.

- Compilation Manager : pyOASIS-extended [DCM_v4.2.1](https://github.com/alexis-barge/DCM/releases/tag/v4.2.1)


### Python

- Eophis version : [v0.9.1](https://github.com/meom-group/eophis/releases/tag/v0.9.1)


### Run

- Production Manager : pyOASIS-extended [DCM_v4.2.1](https://github.com/alexis-barge/DCM/releases/tag/v4.2.1)


### Post-Process

- Post-Process libraries : [DMONTOOLS](https://github.com/alexis-barge/DMONTOOLS) (requires [CDFTOOLS](https://github.com/meom-group/CDFTOOLS))
  
- Plotting : custom scripts in `POSTPROCESS`, with `plots.yml`
