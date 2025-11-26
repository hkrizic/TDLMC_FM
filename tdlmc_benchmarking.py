"""
Benchmarking & plotting helpers for TDLMC inference runs.

This module knows how to:

- Read and analyse multi-start optimisation summaries written by
  `run_multistart` / `run_multistart_vmap`.

- Read the NAUTILUS timing JSON logs created by `run_nautilus`
  (stored under `logs/benchmark_timing` next to the checkpoint).

- Read and analyse EMCEE summaries returned by `run_emcee`
  (optionally saved to JSON with the helpers in this module).

The goal is to extract as much information as possible about:

- Wall-clock times.
- Number of likelihood / log-prob calls.
- Scaling with n_live (NAUTILUS), n_walkers and n_steps (EMCEE).
- CPU / JAX device environment (if present in the summaries/logs).
"""

import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------


def _ensure_ax(ax=None):
    """Return (fig, ax) such that ax is always a valid matplotlib Axes."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _maybe_show_and_save(fig, show: bool = True, savepath: Optional[str] = None):
    """Common logic for show/save."""
    if savepath is not None:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ----------------------------------------------------------------------
# MULTI-START BENCHMARKING
# ----------------------------------------------------------------------


def load_multistart_summary(outdir: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Load a multi-start summary (and best-fit parameters) from disk.

    This is similar to tdlmc_inference.load_multistart_summary, but
    independent of that module so you can use it directly here.

    Parameters
    ----------
    outdir : str
        Directory containing `multi_start_summary.json` and
        `best_fit_params_runXX.json`.

    Returns
    -------
    summary : dict
        Summary with 'best_params_json' filled in if available.
    """
    outdir = os.path.abspath(outdir)
    summary_path = os.path.join(outdir, "multi_start_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"No multi-start summary found at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    best_run = summary.get("best_run", 0)
    best_params_path = os.path.join(outdir, f"best_fit_params_run{best_run:02d}.json")
    if os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        summary["best_params_json"] = best_params

    if verbose:
        print(f"[multistart] Loaded summary from: {summary_path}")
        if "best_params_json" in summary:
            print(f"[multistart] Loaded best-fit parameters from: {best_params_path}")

    return summary


def summarise_multistart(summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute derived statistics from a multi-start summary.

    Returns
    -------
    stats : dict with keys like:
        - n_starts
        - n_param
        - best_loss
        - median_final_loss
        - mean_final_loss
        - final_loss_std
        - chi2_red_median / chi2_red_mean / chi2_red_std
        - total_runtime (if present)
        - env (if present)
    """
    results = summary.get("results", [])
    n_starts = summary.get("n_starts", len(results))
    n_param = summary.get("n_param", None)

    if "final_losses" in summary:
        finals = np.asarray(summary["final_losses"], dtype=float)
    else:
        finals = np.asarray([r["final_loss"] for r in results], dtype=float)

    if "chi2_reds" in summary:
        chi2 = np.asarray(summary["chi2_reds"], dtype=float)
    else:
        chi2 = np.asarray(
            [r.get("chi2_red", np.nan) for r in results], dtype=float
        )

    timing = summary.get("timing", {})
    total_runtime = timing.get("total", None)

    stats = dict(
        n_starts=int(n_starts),
        n_param=int(n_param) if n_param is not None else None,
        best_loss=float(summary.get("best_loss", np.min(finals))),
        median_final_loss=float(np.nanmedian(finals)),
        mean_final_loss=float(np.nanmean(finals)),
        final_loss_std=float(np.nanstd(finals)),
        median_chi2_red=float(np.nanmedian(chi2)) if chi2.size else None,
        mean_chi2_red=float(np.nanmean(chi2)) if chi2.size else None,
        chi2_red_std=float(np.nanstd(chi2)) if chi2.size else None,
        total_runtime=float(total_runtime) if total_runtime is not None else None,
        env=summary.get("env", None),
    )
    return stats


def print_multistart_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print multi-start benchmark statistics."""
    stats = summarise_multistart(summary)
    print("\n=== Multi-start optimisation summary ===")
    print(f"  #starts           : {stats['n_starts']}")
    if stats["n_param"] is not None:
        print(f"  #parameters       : {stats['n_param']}")
    print(f"  best loss         : {stats['best_loss']:.6g}")
    print(
        f"  final loss (median/mean ± std): "
        f"{stats['median_final_loss']:.6g} / "
        f"{stats['mean_final_loss']:.6g} ± {stats['final_loss_std']:.3g}"
    )
    if stats["median_chi2_red"] is not None:
        print(
            f"  chi2_red (median/mean ± std): "
            f"{stats['median_chi2_red']:.3g} / "
            f"{stats['mean_chi2_red']:.3g} ± {stats['chi2_red_std']:.3g}"
        )
    if stats["total_runtime"] is not None:
        print(f"  total runtime     : {stats['total_runtime']:.3f} s")

    env = stats["env"]
    if env is not None:
        print("\n  Environment:")
        print(f"    python          : {env.get('python_version', 'unknown')}")
        print(f"    platform        : {env.get('platform', 'unknown')}")
        print(f"    cpu_count       : {env.get('cpu_count', 'unknown')}")
        print(f"    jax devices     : {env.get('jax_device_count', 'unknown')} "
              f"({', '.join(env.get('jax_platforms', []) or [])})")
        for k, v in env.items():
            if k.startswith("env_"):
                print(f"    {k} = {v}")
        print("========================================\n")


def plot_multistart_best_trace(
    summary: Dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """Plot best-so-far loss as a function of run index."""
    best_trace = np.asarray(summary.get("best_trace", []), dtype=float)
    if best_trace.size == 0:
        raise ValueError("No 'best_trace' found in multi-start summary.")

    fig, ax = _ensure_ax(ax)
    ax.plot(np.arange(len(best_trace)), best_trace, marker="o")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Best-so-far loss")
    ax.set_title("Multi-start: best loss after each run")
    ax.grid(True, alpha=0.3)

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_losses_and_chi2(
    summary: Dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Scatter plot: final loss and reduced chi^2 vs run index.
    """
    results = summary.get("results", [])
    if not results:
        raise ValueError("No 'results' list in multi-start summary.")

    runs = np.asarray([r["run"] for r in results], dtype=int)
    fin = np.asarray([r["final_loss"] for r in results], dtype=float)
    chi2 = np.asarray([r.get("chi2_red", np.nan) for r in results], dtype=float)

    fig, ax = _ensure_ax(ax)
    ax.scatter(runs, fin, label="final loss", marker="o")
    if np.isfinite(chi2).any():
        ax.scatter(runs, chi2, label="reduced chi²", marker="s")
    ax.set_xlabel("Run index")
    ax.set_ylabel("Value")
    ax.set_title("Multi-start: final loss and χ²_red per run")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_timing_breakdown(
    summary: Dict[str, Any],
    ax=None,
    show: bool = True,
    savepath: Optional[str] = None,
):
    """
    Stacked bar plot of timing per run (if available):
    sampling/unconstrain, Adam, L-BFGS-B, I/O.
    """
    results = summary.get("results", [])
    if not results:
        raise ValueError("No 'results' list in multi-start summary.")

    runs = np.asarray([r["run"] for r in results], dtype=int)

    t_sample = np.asarray(
        [r.get("t_sample_unconstrain", 0.0) for r in results], dtype=float
    )
    t_adam = np.asarray([r.get("t_adam", 0.0) for r in results], dtype=float)
    t_lbfgs = np.asarray([r.get("t_lbfgs", 0.0) for r in results], dtype=float)
    t_io = np.asarray([r.get("t_io", 0.0) for r in results], dtype=float)

    fig, ax = _ensure_ax(ax)
    width = 0.8
    ax.bar(runs, t_sample, width, label="sample+unconstrain")
    ax.bar(runs, t_adam, width, bottom=t_sample, label="Adam preopt")
    ax.bar(runs, t_lbfgs, width, bottom=t_sample + t_adam, label="L-BFGS-B")
    ax.bar(
        runs,
        t_io,
        width,
        bottom=t_sample + t_adam + t_lbfgs,
        label="I/O",
    )

    ax.set_xlabel("Run index")
    ax.set_ylabel("Time per run [s]")
    ax.set_title("Multi-start timing breakdown per run")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    _maybe_show_and_save(fig, show=show, savepath=savepath)
    return fig, ax


def plot_multistart_all(
    summary: Dict[str, Any],
    outdir: Optional[str] = None,
    show: bool = True,
):
    """
    Convenience function: produce three standard plots for a multi-start run.

    - best trace vs run index
    - final loss & chi2 vs run index
    - per-run timing breakdown

    If outdir is given, PNG files are saved there.
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        p1 = os.path.join(outdir, "multistart_best_trace.png")
        p2 = os.path.join(outdir, "multistart_losses_chi2.png")
        p3 = os.path.join(outdir, "multistart_timing.png")
    else:
        p1 = p2 = p3 = None

    plot_multistart_best_trace(summary, show=show, savepath=p1)
    plot_multistart_losses_and_chi2(summary, show=show, savepath=p2)
    plot_multistart_timing_breakdown(summary, show=show, savepath=p3)


# ----------------------------------------------------------------------
# NAUTILUS BENCHMARKING (nested sampling)
# ----------------------------------------------------------------------


def load_nautilus_timing_logs(
    path: str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load NAUTILUS timing JSON logs.

    Parameters
    ----------
    path : str
        Either:
          - a NAUTILUS checkpoint file (HDF5), or
          - the 'logs/benchmark_timing' directory itself.

    Returns
    -------
    records : list[dict]
        One dict per timing JSON, with all keys in the "timing" section,
        plus:
            - "checkpoint"
            - "timestamp"
            - "log_path"
    """
    p = Path(path).expanduser().resolve()
    if p.is_file():
        base_dir = p.parent
        ckpt_name = p.stem
        log_dir = base_dir / "logs" / "benchmark_timing"
        prefix = f"timing_{ckpt_name}_"
    else:
        # assume path is a directory containing timing_*.json files
        log_dir = p
        prefix = "timing_"

    if not log_dir.is_dir():
        raise FileNotFoundError(f"No timing log directory found at {log_dir}")

    pattern = str(log_dir / f"{prefix}*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No timing JSON files matching {pattern}")

    records: List[Dict[str, Any]] = []
    for fpath in files:
        with open(fpath, "r") as f:
            payload = json.load(f)

        timing = payload.get("timing", payload)
        rec = dict(timing)
        rec["checkpoint"] = payload.get("checkpoint", None)
        rec["timestamp"] = payload.get("timestamp", None)
        rec["log_path"] = os.path.abspath(fpath)
        records.append(rec)

    # Sort by timestamp if available
    def _key(rec):
        ts = rec.get("timestamp")
        return ts or rec["log_path"]

    records.sort(key=_key)

    if verbose:
        print(f"[nautilus] Loaded {len(records)} timing logs from {log_dir}")
    return records


def summarise_nautilus(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics across multiple NAUTILUS runs.

    Returns
    -------
    stats : dict with keys like:
        - n_runs
        - median_total_runtime
        - median_n_loglike_calls
        - median_loglike_time
        - median_overhead_fraction
        - n_live values, if present
    """
    if not records:
        raise ValueError("Empty records list.")

    total = np.asarray(
        [r.get("total_runtime", np.nan) for r in records], dtype=float
    )
    n_calls = np.asarray(
        [r.get("n_loglike_calls", np.nan) for r in records], dtype=float
    )
    log_tot = np.asarray(
        [r.get("loglike_total", np.nan) for r in records], dtype=float
    )
    overhead = np.asarray(
        [r.get("sampler_overhead", np.nan) for r in records], dtype=float
    )
    n_live = np.asarray(
        [r.get("n_live", np.nan) for r in records], dtype=float
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        overhead_frac = overhead / total

    stats = dict(
        n_runs=len(records),
        median_total_runtime=float(np.nanmedian(total)),
        median_n_loglike_calls=float(np.nanmedian(n_calls)),
        median_loglike_time=float(np.nanmedian(log_tot)),
        median_overhead_fraction=float(np.nanmedian(overhead_frac)),
        n_live_values=n_live.tolist(),
    )
    return stats


def print_nautilus_summary(records: List[Dict[str, Any]]) -> None:
    """Pretty-print summary of NAUTILUS timing logs."""
    stats = summarise_nautilus(records)
    print("\n=== NAUTILUS timing summary ===")
    print(f"  #runs          : {stats['n_runs']}")
    print(f"  total runtime  : median {stats['median_total_runtime']:.3f} s")
    print(f"  n_loglike_calls: median {stats['median_n_loglike_calls']:.3g}")
    print(f"  loglike time   : median {stats['median_loglike_time']:.3f} s")
    print(
        f"  overhead frac  : median {stats['median_overhead_fraction']:.3f} "
        "(sampler overhead / total runtime)"
    )

    # If environment keys are present (from new run_nautilus), print them for
    # the latest record as a representative environment.
    env_keys = [
        "python_version",
        "platform",
        "cpu_count",
        "jax_device_count",
        "jax_platforms",
    ]
    latest = records[-1]
    if any(k in latest for k in env_keys):
        print("\n  Environment (from latest run):")
        for k in env_keys:
            if k in latest:
                print(f"    {k}: {latest[k]}")
        for k, v in latest.items():
            if str(k).startswith("env_"):
                print(f"    {k} = {v}")
    print("================================\n")


def _extract_array(records: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Helper: get float array from a list of dicts."""
    return np.asarray([r.get(key, np.nan) for r in records], dtype=float)


def plot_nautilus_scaling(
    records: List[Dict[str, Any]],
    show: bool = True,
    outdir: Optional[str] = None,
):
    """
    Convenience function: produce several scaling plots for NAUTILUS runs:

    - total runtime vs n_live
    - n_loglike_calls vs n_live
    - average loglike time vs n_live
    - overhead fraction vs n_live
    """
    n_live = _extract_array(records, "n_live")
    total = _extract_array(records, "total_runtime")
    n_calls = _extract_array(records, "n_loglike_calls")
    log_tot = _extract_array(records, "loglike_total")
    overhead = _extract_array(records, "sampler_overhead")

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_call = log_tot / n_calls
        overhead_frac = overhead / total

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # 1) runtime vs n_live
    fig1, ax1 = plt.subplots()
    ax1.scatter(n_live, total)
    ax1.set_xlabel("n_live")
    ax1.set_ylabel("Total runtime [s]")
    ax1.set_title("NAUTILUS: total runtime vs n_live")
    ax1.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig1,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_runtime_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 2) n_loglike_calls vs n_live
    fig2, ax2 = plt.subplots()
    ax2.scatter(n_live, n_calls)
    ax2.set_xlabel("n_live")
    ax2.set_ylabel("n_loglike_calls")
    ax2.set_title("NAUTILUS: likelihood calls vs n_live")
    ax2.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig2,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_calls_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 3) avg loglike time vs n_live
    fig3, ax3 = plt.subplots()
    ax3.scatter(n_live, avg_call)
    ax3.set_xlabel("n_live")
    ax3.set_ylabel("⟨time per loglike call⟩ [s]")
    ax3.set_title("NAUTILUS: mean loglike time vs n_live")
    ax3.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig3,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_avg_loglike_time_vs_nlive.png")
                  if outdir is not None else None),
    )

    # 4) overhead fraction vs n_live
    fig4, ax4 = plt.subplots()
    ax4.scatter(n_live, overhead_frac)
    ax4.set_xlabel("n_live")
    ax4.set_ylabel("Overhead / total runtime")
    ax4.set_title("NAUTILUS: sampler overhead fraction vs n_live")
    ax4.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig4,
        show=show,
        savepath=(os.path.join(outdir, "nautilus_overhead_fraction_vs_nlive.png")
                  if outdir is not None else None),
    )


# ----------------------------------------------------------------------
# EMCEE BENCHMARKING (ensemble MCMC)
# ----------------------------------------------------------------------


def save_emcee_summary(summary: Dict[str, Any], filepath: str) -> str:
    """
    Save an EMCEE summary dictionary to JSON.

    Parameters
    ----------
    summary : dict
        The dict returned by `run_emcee`.
    filepath : str
        Target JSON path.

    Returns
    -------
    filepath : str
        Absolute path to written file.
    """
    filepath = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    return filepath


def load_emcee_summaries(
    paths_or_pattern: Sequence[str] | str,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load one or more EMCEE summary JSON files.

    Parameters
    ----------
    paths_or_pattern : str or sequence of str
        Either:
          - a glob pattern (e.g. 'runs/emcee/*.json'), or
          - an explicit iterable of paths.

    Returns
    -------
    summaries : list[dict]
        List of EMCEE summary dictionaries.
    """
    if isinstance(paths_or_pattern, str):
        files = sorted(glob.glob(paths_or_pattern))
    else:
        files = [str(p) for p in paths_or_pattern]

    if not files:
        raise FileNotFoundError(f"No EMCEE summary files found: {paths_or_pattern}")

    out: List[Dict[str, Any]] = []
    for fpath in files:
        with open(fpath, "r") as f:
            summ = json.load(f)
        summ["_summary_path"] = os.path.abspath(fpath)
        out.append(summ)

    if verbose:
        print(f"[emcee] Loaded {len(out)} summaries.")
    return out


def summarise_emcee(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics over several EMCEE runs.

    Each summary is expected to come from the updated `run_emcee` which
    stores timing info and derived quantities.
    """
    if not summaries:
        raise ValueError("Empty summaries list.")

    def arr(key):
        return np.asarray(
            [s.get(key, np.nan) for s in summaries], dtype=float
        )

    runtime = arr("runtime")
    eff_sps = arr("effective_samples_per_second")
    n_samples = arr("n_samples")
    mean_af = arr("mean_acceptance_fraction")

    stats = dict(
        n_runs=len(summaries),
        median_runtime=float(np.nanmedian(runtime)),
        median_effective_sps=float(np.nanmedian(eff_sps)),
        median_n_samples=float(np.nanmedian(n_samples)),
        median_acceptance_fraction=float(np.nanmedian(mean_af)),
    )
    return stats


def print_emcee_summary(summaries: List[Dict[str, Any]]) -> None:
    """Pretty-print EMCEE benchmark statistics."""
    stats = summarise_emcee(summaries)
    print("\n=== EMCEE benchmark summary ===")
    print(f"  #runs                      : {stats['n_runs']}")
    print(f"  runtime (median)           : {stats['median_runtime']:.3f} s")
    print(f"  n_samples (median)         : {stats['median_n_samples']:.3g}")
    print(
        f"  eff. samples / s (median)  : {stats['median_effective_sps']:.3g}"
    )
    print(
        f"  acceptance fraction (med.) : "
        f"{stats['median_acceptance_fraction']:.3f}"
    )

    latest = summaries[-1]
    env = latest.get("env", None)
    if env:
        print("\n  Environment (from latest run):")
        print(f"    python      : {env.get('python_version', 'unknown')}")
        print(f"    platform    : {env.get('platform', 'unknown')}")
        print(f"    cpu_count   : {env.get('cpu_count', 'unknown')}")
        print(
            f"    jax devices : {env.get('jax_device_count', 'unknown')} "
            f"({', '.join(env.get('jax_platforms', []) or [])})"
        )
        for k, v in env.items():
            if str(k).startswith("env_"):
                print(f"    {k} = {v}")
    print("================================\n")


def plot_emcee_scaling(
    summaries: List[Dict[str, Any]],
    show: bool = True,
    outdir: Optional[str] = None,
):
    """
    Produce standard EMCEE scaling plots:

    - runtime vs total number of samples (n_steps * n_walkers)
    - effective samples per second vs n_walkers
    - acceptance fraction vs n_walkers
    """
    if not summaries:
        raise ValueError("Empty summaries list.")

    n_walkers = np.asarray(
        [s.get("n_walkers", np.nan) for s in summaries], dtype=float
    )
    n_steps = np.asarray(
        [s.get("n_steps", np.nan) for s in summaries], dtype=float
    )
    runtime = np.asarray(
        [s.get("runtime", np.nan) for s in summaries], dtype=float
    )
    eff_sps = np.asarray(
        [s.get("effective_samples_per_second", np.nan) for s in summaries],
        dtype=float,
    )
    af = np.asarray(
        [s.get("mean_acceptance_fraction", np.nan) for s in summaries],
        dtype=float,
    )

    n_samples = n_walkers * n_steps

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # 1) runtime vs total number of samples
    fig1, ax1 = plt.subplots()
    ax1.scatter(n_samples, runtime)
    ax1.set_xlabel("Total samples (n_walkers × n_steps)")
    ax1.set_ylabel("Runtime [s]")
    ax1.set_title("EMCEE: runtime vs total samples")
    ax1.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig1,
        show=show,
        savepath=(os.path.join(outdir, "emcee_runtime_vs_samples.png")
                  if outdir is not None else None),
    )

    # 2) effective samples per second vs n_walkers
    fig2, ax2 = plt.subplots()
    ax2.scatter(n_walkers, eff_sps)
    ax2.set_xlabel("n_walkers")
    ax2.set_ylabel("Effective samples / second")
    ax2.set_title("EMCEE: effective sample rate vs n_walkers")
    ax2.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig2,
        show=show,
        savepath=(os.path.join(outdir, "emcee_eff_sps_vs_nwalkers.png")
                  if outdir is not None else None),
    )

    # 3) acceptance fraction vs n_walkers
    fig3, ax3 = plt.subplots()
    ax3.scatter(n_walkers, af)
    ax3.set_xlabel("n_walkers")
    ax3.set_ylabel("Mean acceptance fraction")
    ax3.set_title("EMCEE: acceptance fraction vs n_walkers")
    ax3.grid(True, alpha=0.3)
    _maybe_show_and_save(
        fig3,
        show=show,
        savepath=(os.path.join(outdir, "emcee_acceptance_vs_nwalkers.png")
                  if outdir is not None else None),
    )

def plot_emcee_diagnostics(
    chain: np.ndarray,
    log_prob_chain: np.ndarray,
    param_names: Sequence[str],
    outdir: Optional[str] = None,
    max_params_to_plot: int = 20,
    prefix: str = "emcee",
    show: bool = True,
):
    """
    Produce standard EMCEE diagnostic plots:
      - Trace plots for first `max_params_to_plot` parameters
      - Log-probability trace
      - Acceptance fraction per walker
      - Corner plot (if `corner` is installed)

    Parameters
    ----------
    chain : array (n_steps, n_walkers, ndim)
    log_prob_chain : array (n_steps, n_walkers)
    param_names : list[str]
    outdir : str or None
    max_params_to_plot : int
    prefix : str
    show : bool
    """

    if chain.ndim != 3:
        raise ValueError("chain must have shape (n_steps, n_walkers, ndim)")

    n_steps, n_walkers, ndim = chain.shape
    n_plot = min(ndim, max_params_to_plot)

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # ----------------------------------------
    # 1) Parameter trace plots
    # ----------------------------------------
    fig, axes = plt.subplots(n_plot, 1, figsize=(8, 2*n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]

    for i in range(n_plot):
        ax = axes[i]
        ax.plot(chain[:, :, i], alpha=0.3)
        ax.set_ylabel(param_names[i])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle("EMCEE parameter traces")

    savepath = (
        os.path.join(outdir, f"{prefix}_trace_plots.png")
        if outdir is not None
        else None
    )
    _maybe_show_and_save(fig, show=show, savepath=savepath)

    # ----------------------------------------
    # 2) Log-probability trace
    # ----------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(log_prob_chain, alpha=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("log probability")
    ax.set_title("EMCEE log-probability trace")
    ax.grid(True, alpha=0.3)

    savepath = (
        os.path.join(outdir, f"{prefix}_logprob_trace.png")
        if outdir is not None
        else None
    )
    _maybe_show_and_save(fig, show=show, savepath=savepath)

    # ----------------------------------------
    # 3) Acceptance fraction per walker
    # ----------------------------------------
    # Fraction of accepted moves = differencing of positions
    diffs = np.diff(chain, axis=0)
    moves = np.any(diffs != 0, axis=2)
    acceptance_fraction = moves.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(n_walkers), acceptance_fraction)
    ax.set_xlabel("Walker index")
    ax.set_ylabel("Acceptance fraction")
    ax.set_title("EMCEE acceptance fraction per walker")
    ax.grid(True, axis="y", alpha=0.3)

    savepath = (
        os.path.join(outdir, f"{prefix}_acceptance_fraction.png")
        if outdir is not None
        else None
    )
    _maybe_show_and_save(fig, show=show, savepath=savepath)

    # ----------------------------------------
    # 4) Corner plot
    # ----------------------------------------
    try:
        import corner

        # Flatten samples across walkers and steps
        flat = chain.reshape(-1, ndim)

        # Only plot subset for clarity
        idxs = list(range(n_plot))
        labels = [param_names[i] for i in idxs]

        fig = corner.corner(flat[:, idxs], labels=labels, show_titles=True)

        savepath = (
            os.path.join(outdir, f"{prefix}_corner.png")
            if outdir is not None
            else None
        )
        _maybe_show_and_save(fig, show=show, savepath=savepath)

    except ImportError:
        print("[emcee diagnostics] corner not installed — skipping corner plot.")



__all__ = [
    # multistart
    "load_multistart_summary",
    "summarise_multistart",
    "print_multistart_summary",
    "plot_multistart_best_trace",
    "plot_multistart_losses_and_chi2",
    "plot_multistart_timing_breakdown",
    "plot_multistart_all",
    # nautilus
    "load_nautilus_timing_logs",
    "summarise_nautilus",
    "print_nautilus_summary",
    "plot_nautilus_scaling",
    # emcee
    "save_emcee_summary",
    "load_emcee_summaries",
    "summarise_emcee",
    "print_emcee_summary",
    "plot_emcee_scaling",
]
