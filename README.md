# TDLMC_FM — Forward Modelling Pipeline for the Time-Delay Lens Modelling Challenge

**TDLMC_FM** is a modular forward-modelling and Bayesian inference pipeline for the [**Time-Delay Lens Modelling Challenge (TDLMC)**](https://tdlmc.github.io).  
It provides a unified framework to perform, benchmark, and visualise probabilistic lens inference using  
[**Nautilus (nested sampling)**](https://nautilus-sampler.readthedocs.io/en/latest/) and [**emcee (MCMC)**](https://emcee.readthedocs.io/en/stable/), built on top of the [**Herculens**](https://github.com/Herculens/herculens) library.


## Repository Structure

```
25NOV_PIPELINE/
├── tdlmc_modelling.ipynb            # Main forward-modelling notebook
│
├── tdlmc_model.py                   # Model definitions & lens setup
├── tdlmc_inference.py               # emcee & Nautilus inference wrappers
├── tdlmc_plotting.py                # Visualisation & diagnostics
├── tdlmc_benchmarking.py            # Runtime and performance tools
├── TDC_util.py                      # Utility functions (file parsing, helpers)
│
├── emcee_output/                    # Stored emcee chains (empty)
├── nautilus_output/                 # Nautilus checkpoints (empty)
├── TDC/                             # Time-Delay Challenge input data
└── TDC_results/                     # Model outputs & analysis results
```
