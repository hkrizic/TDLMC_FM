# 🌌 TDLMC_FM — Forward Modelling Pipeline for the Time-Delay Lens Modelling Challenge

**TDLMC_FM** is a modular forward-modelling and Bayesian inference pipeline for the **Time-Delay Lens Modelling Challenge (TDLMC)**.  
It provides a unified framework to perform, benchmark, and visualise probabilistic lens inference using  
**Nautilus (nested sampling)** and **emcee (MCMC)**, built on top of the **Herculens** library.

---

## ✨ Features

- 🔭 **Forward-Modelling Engine**  
  Built around Herculens for gravitational lens simulation and parameter estimation.

- 🧠 **Unified Bayesian Inference**  
  Consistent interface for **Nautilus** (nested sampling) and **emcee** (MCMC) backends.

- ⚙️ **Modular Design**  
  Clearly separated modules for model setup, inference, plotting, benchmarking, and utilities.

- 🧩 **Configuration-Driven Workflow**  
  Jupyter notebooks read from config sections to ensure reproducibility.

- 📊 **Automated Benchmarking**  
  Evaluate runtime, convergence, and effective sample size across samplers.

- 🎨 **Beautiful Visualisations**  
  Corner plots with truth markers, residual maps, and model-summary panels.

---

## 🧱 Repository Structure

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

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/TDLMC_FM.git
   cd TDLMC_FM
   ```

2. **Create a virtual environment**

   ```bash
   conda create -n tdlmc_env python=3.10
   conda activate tdlmc_env
   pip install -r requirements.txt
   ```

3. **Run the benchmark or modelling notebook**

   ```bash
   jupyter notebook benchmarking_notebook.ipynb
   # or
   jupyter notebook tdlmc_modelling.ipynb
   ```

4. **Outputs**
   - `nautilus_output/` → Nautilus checkpoint HDF5 files  
   - `emcee_output/` → emcee chain backends  
   - `TDC_results/` → diagnostic plots, residuals, and corner plots

---

## 🧮 Example: Comparing Nautilus and emcee

The `benchmarking_notebook.ipynb` allows you to:
- run or load both samplers,
- automatically align posterior outputs,
- generate **joint corner plots**, and
- optionally compute **H₀ posteriors** for cosmological comparison.

---

## 🧠 Dependencies

| Library | Purpose |
|----------|----------|
| **herculens** | Lens modelling and imaging |
| **nautilus-sampler** | Nested sampling |
| **emcee** | MCMC ensemble sampling |
| **getdist** | Corner plots and posterior visualisation |
| **matplotlib / numpy / pandas** | Analysis and plotting |
| **astropy** | FITS and scientific utilities |

---

## 🧩 Citation

If you use **TDLMC_FM** in your work, please cite the original TDLMC papers and the relevant sampler libraries (**Herculens**, **Nautilus**, **emcee**).

---

## 🪐 Author

**Hrvoje Krizic**  
PhD Student, University of Geneva  
📚 Author of *Tutorium Mathematik für Naturwissenschaften* (Springer, 2024)

> *“Forward-modelling where light curves meet likelihoods.”*

---

## 🧰 License

This project is released under the **MIT License**.
