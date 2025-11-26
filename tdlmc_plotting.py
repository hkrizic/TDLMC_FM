# tdlmc_plotting.py
"""
Plotting and analysis helpers for the TDLMC lens modelling workflow.

This module is purely about *visualisation* and simple scalar diagnostics.
It assumes model/inference objects are already built elsewhere.

Provided functions:

- plot_multistart_history(...)
- plot_bestfit_model(...)
- nautilus_corner_plot(...)
- nautilus_mean_model_plot(...)
"""

import os
from typing import Dict, Sequence, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from getdist import plots, MCSamples
from TDC_util import parse_lens_info_file


def plot_multistart_history(
    best_trace: Sequence[float],
    final_losses: Sequence[float],
    outdir: Optional[str] = None,
    rel_eps: float = 0.01,
    filename: str = "best_loss_vs_run.png",
    chi2_given: bool = False,
):
    """
    Plot best-so-far safe loss vs. multi-start run, with per-run final losses overlaid.

    Parameters
    ----------
    best_trace : sequence of float
        Best safe-loss value after each multi-start run.
    final_losses : sequence of float
        Final safe-loss of each individual run.
    outdir : str or None
        If given, the figure is saved as `filename` inside this directory.
    rel_eps : float
        If >0, vertical line is drawn at the last run where the best loss
        improved by more than `rel_eps` relative to the previous best.
    """
    best_trace = np.asarray(best_trace, float)
    final_losses = np.asarray(final_losses, float)
    runs = np.arange(len(best_trace))

    plt.figure(figsize=(7, 4.5))
    if chi2_given:
        plt.step(runs, best_trace, where="post", label="Best-so-far (reduced chi²)")
        plt.scatter(
            runs, final_losses, s=16, alpha=0.6, label="Per-run final reduced chi²"
        )
    else:
        plt.step(runs, best_trace, where="post", label="Best-so-far (safe loss)")
        plt.scatter(runs, final_losses, s=16, alpha=0.6, label="Per-run final loss")

    # mark last significant improvement
    if rel_eps is not None and len(best_trace) > 1:
        improvements = best_trace[:-1] - best_trace[1:]
        rel_impr = improvements / np.maximum(1e-12, best_trace[:-1])
        sig_idxs = np.where(rel_impr > rel_eps)[0]
        if sig_idxs.size > 0:
            last_sig = int(sig_idxs[-1] + 1)
            plt.axvline(
                last_sig,
                linestyle="--",
                alpha=0.5,
                label=f"Last >{int(rel_eps*100)}% gain @ run {last_sig}",
            )
    if chi2_given:
        plt.xlabel("Run #")
        plt.ylabel(r"Reduced $\chi^2$")
        plt.title(r"Best-so-far reduced $\chi^2$ vs. multi-start run")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
    else:
        plt.xlabel("Run #")
        plt.ylabel("Safe loss")
        plt.title("Best-so-far loss vs. multi-start run")
        plt.grid(True, alpha=0.25)
        plt.legend(loc="best")
    plt.tight_layout()

    if outdir is not None:
        path = os.path.join(outdir, filename)
        plt.savefig(path, dpi=150)
        print(f"Saved multi-start history plot to: {path}")

    plt.show()

def plot_bestfit_model(
    prob_model,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    params: Dict,
    outdir: Optional[str] = None,
    tag: str = "bestfit",
    plotherculens: bool = True,
    plot_ownresiduals: bool = False,
    print_chi2: bool = True,
):
    """
    Plot a best-fit model summary and basic diagnostics (data / model / residuals).

    Parameters
    ----------
    prob_model : tdlmc_model.ProbModel
        Model instance providing `params2kwargs`.
    lens_image : herculens.LensImage
    img, noise_map : np.ndarray
    plotter : herculens.Analysis.plot.Plotter
    params : dict
        Constrained parameter dict (e.g. from `prob_model.constrain(u)` or from
        the JSON files produced by `run_multistart`).
    """
    kwargs_best = prob_model.params2kwargs(params)
    model_img = lens_image.model(**kwargs_best)
    resid = (img - model_img) / (noise_map + 1e-12)
    if plotherculens:
        # Heruclens summary (data, model, residuals + source plane)
        fig = plotter.model_summary(
            lens_image,
            kwargs_best,
            show_source=True,
            kwargs_grid_source=dict(pixel_scale_factor=1),
        )
        if outdir is not None:
            diag_path = os.path.join(outdir, f"{tag}_herculens_diagnostics.png")
            plt.savefig(diag_path, dpi=200, bbox_inches="tight")
            print(f"Saved best-fit herculens-diagnostics to: {diag_path}")
        plt.show()
    if plot_ownresiduals:
        # Data / model / residuals (S/N)
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        im0 = ax[0].imshow(img, origin="lower", cmap="afmhot")
        ax[0].set_title("Data")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

        im1 = ax[1].imshow(model_img, origin="lower", cmap="afmhot")
        ax[1].set_title("Model")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

        im2 = ax[2].imshow(resid, origin="lower", cmap="bwr", vmin=-5, vmax=5)
        ax[2].set_title("Residuals (S/N)")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

        plt.tight_layout()

        if outdir is not None:
            diag_path = os.path.join(outdir, f"{tag}_own_diagnostics.png")
            plt.savefig(diag_path, dpi=200, bbox_inches="tight")
            print(f"Saved best-fit own-diagnostics to: {diag_path}")

        plt.show()

    chi2 = float(np.sum(resid**2))
    n_pix = img.size
    n_param = prob_model.num_parameters
    chi2_red = chi2 / max(1, (n_pix - n_param))
    if print_chi2:
        print(f"reduced chi^2 = {chi2_red:.3f}")


def nautilus_corner_plot(
    prior,
    loglike,
    base: str,
    rung: int,
    code_id: int,
    seed: int,
    number_live: int,
    params_to_corner: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    rel_error: bool = False,
):
    """
    Make a GetDist corner plot from a NAUTILUS posterior, with TDLMC truth markers.

    Parameters
    ----------
    prior, loglike : as returned by `tdlmc_inference.build_nautilus_prior_and_loglike`
    base, rung, code_id, seed : identify the TDLMC system
    number_live : int
        Number of live points used in the NAUTILUS run (needed to find the checkpoint).
    params_to_corner : list of str or None
        Parameters to include in the corner plot. If None, a default set is used.
    out_dir : str or None
        Directory where the corner plot PNG will be written. If None, a default
        nautilus_output/... path is used.
    """
    from tdlmc_inference import load_posterior_from_checkpoint

    if params_to_corner is None:
        params_to_corner = [
            "lens_theta_E",
            "lens_gamma",
            "lens_e1",
            "lens_e2",
            "light_Re_L",
            "light_n_L",
            "light_e1_L",
            "light_e2_L",
        ]

    if out_dir is None:
        out_dir = os.path.join(
            "nautilus_output",
            f"rung{rung}",
            f"code{code_id}",
            f"f160w-seed{seed}",
        )
    os.makedirs(out_dir, exist_ok=True)

    ckpt = os.path.join(
        "nautilus_output",
        f"run_checkpoint_rung{rung}_seed{seed}_{number_live}.hdf5",
    )

    sampler, points, log_w, log_l = load_posterior_from_checkpoint(
        prior, loglike, n_live=number_live, filepath=ckpt
    )
    weights = np.exp(log_w)
    df = pd.DataFrame(points, columns=sampler.prior.keys)

    available = [p for p in params_to_corner if p in df.columns]
    samples_list = [df[p].to_numpy() for p in available]
    names = available

    # --- Truth values from open-box file ---
    code = f"code{code_id}"
    truth_file = os.path.join(
        base,
        f"TDC/rung{rung}_open_box/{code}/f160w-seed{seed}/lens_all_info.txt",
    )
    lens_info = parse_lens_info_file(truth_file)

    def _phi_to_rad(phi):
        return np.deg2rad(phi) if np.abs(phi) > 2 * np.pi else float(phi)

    def _e1e2_from_q_phi(q, phi):
        e = (1 - q) / (1 + q)
        return e * np.cos(2 * phi), e * np.sin(2 * phi)

    # lens mass truth
    thetaE_true = lens_info["lens_mass_model"]["SPEMD"]["theta_E"]
    gamma_true = lens_info["lens_mass_model"]["SPEMD"]["gamma"]
    q_mass = lens_info["lens_mass_model"]["SPEMD"]["q"]
    phi_mass = _phi_to_rad(
        lens_info["lens_mass_model"]["SPEMD"].get("phi_G", 0.0)
    )
    e1_mass_true, e2_mass_true = _e1e2_from_q_phi(q_mass, phi_mass)

    # lens light truth
    q_light = lens_info["lens_light"]["q"]
    phi_light = _phi_to_rad(lens_info["lens_light"]["phi_G"])
    e1_L_true, e2_L_true = _e1e2_from_q_phi(q_light, phi_light)
    R_true = lens_info["lens_light"]["R_sersic"]
    n_true = lens_info["lens_light"]["n_sersic"]

    truth_values = {
        "lens_theta_E": thetaE_true,
        "lens_gamma": gamma_true,
        "lens_e1": e1_mass_true,
        "lens_e2": e2_mass_true,
        "light_Re_L": R_true,
        "light_n_L": n_true,
        "light_e1_L": e1_L_true,
        "light_e2_L": e2_L_true,
    }
    markers = [truth_values.get(k, np.nan) for k in names]

    # Build GetDist MCSamples
    settings_mcsamples = {
        "smooth_scale_1D": 0.5,
        "smooth_scale_2D": 0.5,
    }
    mcsamples_nautilus = MCSamples(
        samples=samples_list, names=names, settings=settings_mcsamples, weights=weights
    )

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.legend_fontsize = 18
    g.settings.axes_labelsize = 14

    mcsamples_list = [mcsamples_nautilus]
    colors = ["tab:blue"]
    contour_lws = [2]
    legend_labels = ["Posterior (NAUTILUS)"]

    g.triangle_plot(
        mcsamples_list,
        params=names,
        legend_labels=legend_labels,
        filled=True,
        colors=colors,
        contour_colors=colors,
        markers=markers,
        contour_lws=contour_lws,
    )

    # uniform rescale of axes (so contours have some breathing room)
    scale_factor = 1.1
    n = len(names)
    for i in range(n):
        ax_diag = g.subplots[i][i]
        x_lo, x_hi = ax_diag.get_xlim()
        x_mid = 0.5 * (x_hi + x_lo)
        x_half = 0.5 * (x_hi - x_lo) * scale_factor
        ax_diag.set_xlim(x_mid - x_half, x_mid + x_half)
        for j in range(i):
            ax = g.subplots[i][j]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_mid = 0.5 * (x_hi + x_lo)
            y_mid = 0.5 * (y_hi + y_lo)
            ax.set_xlim(
                x_mid - 0.5 * (x_hi - x_lo) * scale_factor,
                x_mid + 0.5 * (x_hi - x_lo) * scale_factor,
            )
            ax.set_ylim(
                y_mid - 0.5 * (y_hi - y_lo) * scale_factor,
                y_mid + 0.5 * (y_hi - y_lo) * scale_factor,
            )

    # Add truth values + % error or Δ on diagonals
    def _weighted_mean(x, w):
        w = np.asarray(w, float)
        w = w / (w.sum() + 1e-300)
        return float(np.sum(w * np.asarray(x, float)))

    for i, name in enumerate(names):
        mu = _weighted_mean(samples_list[i], weights)
        tv = truth_values.get(name, np.nan)
        ax = g.subplots[i][i]

        if rel_error and np.isfinite(tv) and abs(tv) > 1e-12:
            err_pct = 100.0 * (mu - tv) / tv
            label = f"$\\Delta\\%= {err_pct:+.2f}\\%$"
        else:
            err_abs = mu - tv
            label = (
                f"$\\mathrm{{Truth}} = {tv:.3f}$\n"
                f"$\\Delta = {err_abs:+.2e}$"
            )

        ax.text(
            0.98,
            0.96,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85
            ),
        )

    out_path = os.path.join(
        out_dir, f"corner_nautilus_rung{rung}_seed{seed}.png"
    )
    g.export(out_path, dpi=300)
    plt.show()
    print("Saved NAUTILUS corner plot to:", out_path)


def nautilus_mean_model_plot(
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    prior,
    loglike,
    paramdict_to_kwargs,
    rung: int,
    code_id: int,
    seed: int,
    number_live: int,
    out_dir: Optional[str] = None,
):
    """
    Plot the NAUTILUS posterior mean model and residuals.

    Parameters
    ----------
    lens_image, img, noise_map, plotter : as usual
    prior, loglike, paramdict_to_kwargs : from `build_nautilus_prior_and_loglike`
    rung, code_id, seed, number_live : identify the run / checkpoint.
    """
    from tdlmc_inference import load_posterior_from_checkpoint

    if out_dir is None:
        out_dir = os.path.join(
            "nautilus_output",
            f"rung{rung}",
            f"code{code_id}",
            f"f160w-seed{seed}",
        )
    os.makedirs(out_dir, exist_ok=True)

    ckpt = os.path.join(
        "nautilus_output",
        f"run_checkpoint_rung{rung}_seed{seed}_{number_live}.hdf5",
    )

    sampler, points, log_w, log_l = load_posterior_from_checkpoint(
        prior, loglike, n_live=number_live, filepath=ckpt
    )
    weights = np.exp(log_w)
    cols = sampler.prior.keys
    df = pd.DataFrame(points, columns=cols)

    def wmean(x, w):
        w = np.asarray(w, float)
        w /= (w.sum() + 1e-300)
        return float(np.dot(np.asarray(x, float), w))

    mean_sample = {k: wmean(df[k].values, weights) for k in cols}

    kwargs_mean = paramdict_to_kwargs(mean_sample)

    # Plot summary (data, model, residuals + source)
    fig = plotter.model_summary(
        lens_image,
        kwargs_mean,
        show_source=True,
        kwargs_grid_source=dict(pixel_scale_factor=1),
    )
    out_summary = os.path.join(
        out_dir,
        f"nautilus_mean_model_summary_rung{rung}_seed{seed}_nlive{number_live}.png",
    )
    plt.savefig(out_summary, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved mean-model summary to:", out_summary)

    model_img = lens_image.model(**kwargs_mean)
    resid = (img - model_img) / (noise_map + 1e-12)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    im0 = ax[0].imshow(img, origin="lower", cmap="afmhot")
    ax[0].set_title("Data")
    plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(model_img, origin="lower", cmap="afmhot")
    ax[1].set_title("Mean model")
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(resid, origin="lower", cmap="bwr", vmin=-5, vmax=5)
    ax[2].set_title("Residuals (S/N)")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_diag = os.path.join(
        out_dir,
        f"nautilus_mean_model_diagnostics_rung{rung}_seed{seed}_nlive{number_live}.png",
    )
    plt.savefig(out_diag, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved NAUTILUS diagnostics to:", out_diag)

    chi2 = float(np.sum(resid**2))
    n_pix = img.size
    n_params_eff = len(cols)
    chi2_red = chi2 / max(1, (n_pix - n_params_eff))
    print(f"chi^2 = {chi2:.2f}   |   chi^2_red = {chi2_red:.3f}")



def pso_corner_plot(
    base: str,
    rung: int,
    code_id: int,
    seed: int,
    out_dir: str,
    chain_path: Optional[str] = None,
    params_to_corner: Optional[List[str]] = None,
    rel_error: bool = False,
):
    """
    Make a GetDist corner plot from the PSO swarm samples, with TDLMC truth markers.

    This is analogous to `nautilus_corner_plot`, but the samples come from
    the PSO swarm (`pso_chain.npz`) instead of a nested-sampling posterior.

    Parameters
    ----------
    base, rung, code_id, seed : identify the TDLMC system.
    out_dir : str
        Directory where PSO results are stored (including `pso_chain.npz`).
    chain_path : str or None
        Path to the PSO chain file (default = <out_dir>/pso_chain.npz).
    params_to_corner : list of str or None
        Parameters to include in the corner plot. If None, a default set is used.
    rel_error : bool
        If True, show relative error in the diagonal annotations.
    """
    if params_to_corner is None:
        params_to_corner = [
            "lens_theta_E",
            "lens_gamma",
            "lens_e1",
            "lens_e2",
            "light_Re_L",
            "light_n_L",
            "light_e1_L",
            "light_e2_L",
        ]

    if chain_path is None:
        chain_path = os.path.join(out_dir, "pso_chain.npz")

    if not os.path.exists(chain_path):
        raise FileNotFoundError(
            f"PSO chain file not found at {chain_path}.\n"
            "Re-run run_pso(..., save_chain=True) to create it."
        )

    data = np.load(chain_path, allow_pickle=True)
    samples_all = np.asarray(data["samples"], float)  # (n_samples, ndim)
    names_all = [str(n) for n in data["names"]]

    # Map param -> column index
    name_to_idx = {n: i for i, n in enumerate(names_all)}

    # Select only requested / available params
    available = [p for p in params_to_corner if p in name_to_idx]
    if not available:
        raise ValueError(
            "None of the requested params_to_corner are in the PSO chain file."
        )
    names = available
    samples_list = [samples_all[:, name_to_idx[p]] for p in names]

    # --- Truth values from open-box file (same as nautilus_corner_plot) ---
    code = f"code{code_id}"
    truth_file = os.path.join(
        base,
        f"TDC/rung{rung}_open_box/{code}/f160w-seed{seed}/lens_all_info.txt",
    )
    lens_info = parse_lens_info_file(truth_file)

    def _phi_to_rad(phi):
        return np.deg2rad(phi) if np.abs(phi) > 2 * np.pi else float(phi)

    def _e1e2_from_q_phi(q, phi):
        e = (1 - q) / (1 + q)
        return e * np.cos(2 * phi), e * np.sin(2 * phi)

    # lens mass truth
    thetaE_true = lens_info["lens_mass_model"]["SPEMD"]["theta_E"]
    gamma_true = lens_info["lens_mass_model"]["SPEMD"]["gamma"]
    q_mass = lens_info["lens_mass_model"]["SPEMD"]["q"]
    phi_mass = _phi_to_rad(
        lens_info["lens_mass_model"]["SPEMD"].get("phi_G", 0.0)
    )
    e1_mass_true, e2_mass_true = _e1e2_from_q_phi(q_mass, phi_mass)

    # lens light truth
    q_light = lens_info["lens_light"]["q"]
    phi_light = _phi_to_rad(lens_info["lens_light"]["phi_G"])
    e1_L_true, e2_L_true = _e1e2_from_q_phi(q_light, phi_light)
    R_true = lens_info["lens_light"]["R_sersic"]
    n_true = lens_info["lens_light"]["n_sersic"]

    truth_values = {
        "lens_theta_E": thetaE_true,
        "lens_gamma": gamma_true,
        "lens_e1": e1_mass_true,
        "lens_e2": e2_mass_true,
        "light_Re_L": R_true,
        "light_n_L": n_true,
        "light_e1_L": e1_L_true,
        "light_e2_L": e2_L_true,
    }
    markers = [truth_values.get(k, np.nan) for k in names]

    # --- Build GetDist MCSamples (same style as NAUTILUS) ---
    settings_mcsamples = {
        "smooth_scale_1D": 0.5,
        "smooth_scale_2D": 0.5,
    }
    # equal weights for swarm samples
    weights = np.ones(samples_all.shape[0], dtype=float)

    mcsamples_pso = MCSamples(
        samples=samples_list, names=names, settings=settings_mcsamples, weights=weights
    )

    g = plots.get_subplot_plotter(subplot_size=2)
    g.settings.legend_fontsize = 18
    g.settings.axes_labelsize = 14

    g.triangle_plot(
        [mcsamples_pso],
        params=names,
        legend_labels=["PSO swarm"],
        filled=True,
        colors=["tab:orange"],
        contour_colors=["tab:orange"],
        contour_lws=[2],
        markers=markers,
    )

    # uniform rescale of axes (so contours have some breathing room)
    scale_factor = 1.1
    n = len(names)
    for i in range(n):
        ax_diag = g.subplots[i][i]
        x_lo, x_hi = ax_diag.get_xlim()
        x_mid = 0.5 * (x_hi + x_lo)
        x_half = 0.5 * (x_hi - x_lo) * scale_factor
        ax_diag.set_xlim(x_mid - x_half, x_mid + x_half)
        for j in range(i):
            ax = g.subplots[i][j]
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            x_mid = 0.5 * (x_hi + x_lo)
            y_mid = 0.5 * (y_hi + y_lo)
            ax.set_xlim(
                x_mid - 0.5 * (x_hi - x_lo) * scale_factor,
                x_mid + 0.5 * (x_hi - x_lo) * scale_factor,
            )
            ax.set_ylim(
                y_mid - 0.5 * (y_hi - y_lo) * scale_factor,
                y_mid + 0.5 * (y_hi - y_lo) * scale_factor,
            )

    # Diagonal annotations: compare swarm mean to truth
    def _weighted_mean(x, w):
        w = np.asarray(w, float)
        w = w / (w.sum() + 1e-300)
        return float(np.sum(w * np.asarray(x, float)))

    for i, name in enumerate(names):
        mu = _weighted_mean(samples_list[i], weights)
        tv = truth_values.get(name, np.nan)
        ax = g.subplots[i][i]

        if rel_error and np.isfinite(tv) and abs(tv) > 1e-12:
            err_pct = 100.0 * (mu - tv) / tv
            label = f"$\\Delta\\%= {err_pct:+.2f}\\%$"
        else:
            err_abs = mu - tv
            label = (
                f"$\\mathrm{{Truth}} = {tv:.3f}$\n"
                f"$\\Delta = {err_abs:+.2e}$"
            )

        ax.text(
            0.98,
            0.96,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round,pad=0.18", fc="white", ec="0.6", alpha=0.85
            ),
        )

    out_path = os.path.join(out_dir, f"corner_pso_rung{rung}_seed{seed}.png")
    g.export(out_path, dpi=300)
    plt.show()
    print("Saved PSO corner plot to:", out_path)


def pso_best_model_plot(
    prob_model,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
    plotter,
    best_params_json: Dict,
    out_dir: Optional[str] = None,
    tag: str = "pso_best",
    plotherculens: bool = True,
    plot_ownresiduals: bool = False,
    print_chi2: bool = True,
):
    """
    Plot data / PSO best-fit model / residuals using the same machinery
    as `plot_bestfit_model`.

    `best_params_json` may be either:
      - a dict with 'x_image', 'y_image', 'ps_amp' arrays (multistart-style), or
      - a flattened dict with 'x_image_0', 'y_image_0', 'ps_amp_0', ...
        (PSO / NAUTILUS-style).

    This function:
      1. Unflattens any 'x_image_i' etc. into arrays if needed.
      2. Calls `plot_bestfit_model(...)` with the resulting parameter dict.
    """

    # Start from a copy so we don't mutate the caller's dict
    params = dict(best_params_json)

    # Detect if we have flattened point-source keys
    has_flat_ps = any(k.startswith("x_image_") for k in params) and "x_image" not in params

    if has_flat_ps:
        # collect all indices i from keys like 'x_image_i'
        indices = sorted(
            {
                int(k.split("_")[-1])
                for k in params.keys()
                if k.startswith("x_image_")
            }
        )
        if indices:
            x_im = np.array([params[f"x_image_{i}"] for i in indices])
            y_im = np.array([params[f"y_image_{i}"] for i in indices])
            amp_im = np.array([params[f"ps_amp_{i}"] for i in indices])

            params["x_image"] = x_im
            params["y_image"] = y_im
            params["ps_amp"] = amp_im

            # remove flattened keys; `params2kwargs` expects the array versions
            for i in indices:
                params.pop(f"x_image_{i}", None)
                params.pop(f"y_image_{i}", None)
                params.pop(f"ps_amp_{i}", None)

    # Now just use the same plotting as for a multistart best-fit
    return plot_bestfit_model(
        prob_model=prob_model,
        lens_image=lens_image,
        img=img,
        noise_map=noise_map,
        plotter=plotter,
        params=params,
        outdir=out_dir,
        tag=tag,
        plotherculens=plotherculens,
        plot_ownresiduals=plot_ownresiduals,
        print_chi2=print_chi2,
    )


__all__ = [
    "plot_multistart_history",
    "plot_bestfit_model",
    "nautilus_corner_plot",
    "nautilus_mean_model_plot",
    "pso_corner_plot",
    "pso_best_model_plot",
]
