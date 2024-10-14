# MLE-Fluxes

[![DOI](https://zenodo.org/badge/763681074.svg)](https://doi.org/10.5281/zenodo.13851909)

## Context and Motivation

Purpose of this experiment is to compute the vertical buoyancy fluxes induced by different submesoscale Mixed Layer Eddies (MLE) parameterisation on a 10 years global eORCA025 config.
Internal and external computed fluxes are written in an output file with the NEMO output system (XIOS).

#### Variations
- **STD** : External standard computation, as described in [Calvert et al. (2020)](https://doi.org/10.1016/j.ocemod.2020.101678) with retroaction on the solution
- **CNN** : Fluxes computed from pre-trained [Bodner, Balwada and Zanna (2024)]() CNN

In reality, four experiments are realized :
- `eORCA025.L75` : no MLE, used as reference to compare the effects of the different methods on mixed layer depth
- `eORCA05.L75-MLE.STD-NEMO` : Computation of [Calvert et al. (2020)](https://doi.org/10.1016/j.ocemod.2020.101678) with the NEMO implementation
- `eORCA05.L75-MLE.STD-Python` : corresponds to **STD** variation
- `eORCA05.L75-MLE.CNN` : corresponds to **CNN** variation

<img width="695" alt="MLE_EXP" src="https://github.com/morays-community/NEMO-MLE_Fluxes/assets/138531178/084171b2-7f5d-407b-ad6c-92551f3bbcb2">

## Experiments Requirements


### Compilation

- NEMO version : [v4.2.1](https://forge.nemo-ocean.eu/nemo/nemo/-/releases/4.2.1) patched with [morays](https://github.com/morays-community/Patches-NEMO/tree/main/NEMO_v4.2.1) and local `CONFIG/src` sources.

- Compilation Manager : pyOASIS-extended [DCM_v4.2.1](https://github.com/alexis-barge/DCM/releases/tag/v4.2.1)


### Python

- Eophis version : [v1.0.0](https://github.com/meom-group/eophis/releases/tag/v1.0.0)
- **CNN** dependencies:
  ```bash
    git submodule update --init --recursive
    cd MLE-Fluxes.CNN/INFERENCES/NEMO_MLE
    pip install -e .  
  ```

### Run

- Production Manager : pyOASIS-extended [DCM_v4.2.1](https://github.com/alexis-barge/DCM/releases/tag/v4.2.1)


### Post-Process

- Post-Process libraries : [DMONTOOLS](https://github.com/alexis-barge/DMONTOOLS) (requires [CDFTOOLS](https://github.com/meom-group/CDFTOOLS))
  
- Plotting : custom scripts in `POSTPROCESS`, with `plots.yml`
