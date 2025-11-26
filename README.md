# TDLMC_FM — Forward Modelling Pipeline for the Time-Delay Lens Modelling Challenge

**TDLMC_FM** is a modular forward-modelling and Bayesian inference pipeline for the **Time-Delay Lens Modelling Challenge (TDLMC)**.  
It provides a unified framework to perform, benchmark, and visualise probabilistic lens inference using  
**Nautilus (nested sampling)** and **emcee (MCMC)**, built on top of the **Herculens** library.


## Repository Structure

```
25NOV_PIPELINE/
├── benchmarking_notebook.ipynb      # Unified benchmark + comparison notebook
├── tdlmc_modelling.ipynb            # Main forward-modelling notebook
│
├── tdlmc_model.py                   # Model definitions & lens setup
├── tdlmc_inference.py               # emcee & Nautilus inference wrappers
├── tdlmc_plotting.py                # Visualisation & diagnostics
├── tdlmc_benchmarking.py            # Runtime and performance tools
├── TDC_util.py                      # Utility functions (file parsing, helpers)
│
├── emcee_output/                    # Stored emcee chains
├── nautilus_output/                 # Nautilus checkpoints
├── TDC/                             # Time-Delay Challenge input data
└── TDC_results/                     # Model outputs & analysis results
```


## License

This project is released under the **MIT License**.
