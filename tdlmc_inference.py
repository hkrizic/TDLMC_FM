# tdlmc_inference.py
"""
Inference utilities for the TDLMC lens model.

This module contains:
- multi-start optimisation (Adam preopt + L-BFGS-B) around a herculens+NumPyro ProbModel
- Nautilus nested sampling helper functions (prior + likelihood + runner).
- SVI chain sampler using NumPyro's AutoLowRankMultivariateNormal guide.
- EMC / emcee ensemble MCMC helpers for comparison with Nautilus and SVI.
"""
import os
import time
import json
from typing import Dict, Tuple
import jaxopt  # NEW: pure-JAX optimizers (LBFGS)

from tqdm import trange

import numpy as np
import jax
import jax.numpy as jnp
import optax
from numpyro import infer
from numpyro.infer import autoguide
from numpyro.infer.util import unconstrain_fn

from herculens.Inference.loss import Loss
from herculens.Inference.Optimization.jaxopt import JaxoptOptimizer

from nautilus import Prior, Sampler
from scipy.stats import norm, truncnorm, lognorm
import emcee


import platform
import socket
import getpass
from datetime import datetime



def get_environment_info() -> Dict[str, object]:
    """
    Collect basic environment information for benchmarking.

    This is attached to multi-start summaries, NAUTILUS timing logs
    and EMCEE summaries so that scaling with CPU / threads / devices
    can be analysed afterwards.
    """
    info: Dict[str, object] = {}

    # Basic system / process info
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    info["hostname"] = socket.gethostname()
    try:
        info["user"] = getpass.getuser()
    except Exception:
        pass
    info["pid"] = os.getpid()
    info["cpu_count"] = os.cpu_count()

    # Library versions
    try:
        info["numpy_version"] = np.__version__
    except Exception:
        pass
    try:
        info["jax_version"] = jax.__version__
    except Exception:
        pass

    # JAX devices
    try:
        devices = jax.devices()
        info["jax_device_count"] = len(devices)
        info["jax_platforms"] = sorted({d.platform for d in devices})
    except Exception:
        pass

    # Thread / BLAS / JAX env variables (useful for CPU scaling)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "JAX_NUM_THREADS",
    ):
        if var in os.environ:
            info[f"env_{var.lower()}"] = os.environ[var]

    # UTC timestamp for the *start* of the run
    info["timestamp"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    return info



# ----------------------------------------------------------------------
# Small helper for high-resolution timing
# ----------------------------------------------------------------------

def now() -> float:
    """Return a high-resolution wall-clock time (seconds)."""
    return time.perf_counter()


def _nautilus_points_to_array(points):
    """
    Convert NAUTILUS 'points' into (samples, param_names).

    If 'points' is a structured array, returns:
        samples : (n_samples, ndim) float
        param_names : list of field names

    Otherwise assumes 'points' is already an array of shape (n_samples, ndim)
    and returns (np.asarray(points), None).
    """
    if hasattr(points, "dtype") and points.dtype.names is not None:
        names = list(points.dtype.names)
        samples = np.vstack([np.asarray(points[name], float) for name in names]).T
        return samples, names
    else:
        arr = np.asarray(points, float)
        return arr, None


def _get_rng(random_seed):
    """
    Helper to obtain a numpy RandomState.

    If random_seed is None, returns the global np.random module.
    Otherwise returns a new RandomState(random_seed).
    """
    if random_seed is None:
        return np.random
    return np.random.RandomState(int(random_seed))


def _standard_posterior_dict(engine, samples, log_likelihood, log_weights,
                             param_names=None, extra=None):
    """
    Build a standardised posterior dictionary.

    Parameters
    ----------
    engine : str
        "nautilus" or "emcee".
    samples : array-like, shape (n_samples, ndim)
    log_likelihood : array-like or None
    log_weights : array-like or None
    param_names : list[str] or None
    extra : dict or None
        Extra metadata (n_raw_samples, timing, etc.).
    """
    samples = np.asarray(samples, float)
    if log_likelihood is not None:
        log_likelihood = np.asarray(log_likelihood, float)
    if log_weights is not None:
        log_weights = np.asarray(log_weights, float)
    if param_names is not None:
        param_names = list(param_names)

    return dict(
        engine=str(engine),
        samples=samples,
        log_likelihood=log_likelihood,
        log_weights=log_weights,
        param_names=param_names,
        meta=extra or {},
    )




def make_safe_loss(loss_obj):
    """Wrap a herculens Loss object so that non-finite values are replaced by a large constant."""
    def _safe(uvec):
        val = loss_obj(uvec)
        return jnp.where(jnp.isfinite(val), val, jnp.array(1e30))
    return _safe  # new safer loss function


def adam_preopt(loss_fn, u0, n_steps: int = 200, lr: float = 5e-3):
    """
    Short Adam pre-optimisation of an unconstrained parameter vector `u0`.
    Returns the best point found during the Adam loop.
    """

    # Initialize optimizer
    opt = optax.adam(lr)
    state = opt.init(u0)

    # Single Adam step
    @jax.jit
    def step(u, state):
        l, g = jax.value_and_grad(loss_fn)(u)
        updates, state = opt.update(g, state)
        u2 = optax.apply_updates(u, updates)
        return u2, state, l

    u = u0
    best = u0
    best_val = float(loss_fn(u0))
    for _ in range(n_steps):
        u, state, l = step(u, state)
        lv = float(l)
        if np.isfinite(lv) and lv < best_val:
            best = u
            best_val = lv
    return best


def run_multistart(
    prob_model,
    img: np.ndarray,
    noise_map: np.ndarray,
    outdir: str,
    n_starts: int = 20,
    random_seed: int = 73,
    do_preopt: bool = True,
    adam_steps: int = 250,
    adam_lr: float = 3e-3,
    maxiter: int = 600,
    rel_eps: float = 0.01,
    verbose: bool = False,
) -> Dict:
    """
    Multi-start optimisation using samples from the NumPyro prior as starting points.

    Parameters
    ----------
    prob_model : tdlmc_model.ProbModel
        Probabilistic model instance.
    img, noise_map : np.ndarray
        Data and per-pixel noise (same shapes as used to build prob_model).
    outdir : str
        Output directory where JSON files and summary will be written.
    n_starts : int
        Number of random starting points.
    random_seed : int
        Base seed for generating starting points.
    do_preopt : bool
        If True, run a short Adam pre-optimisation before L-BFGS-B.

    Returns
    -------
    summary : dict
        Dictionary containing:
          - "results": list of per-run dicts (run, init_loss, pre_loss, final_loss,
            runtime, t_sample_unconstrain, t_adam, t_lbfgs, t_io, is_best,
            chi2_red)
          - "best_run": index of the best run
          - "best_loss": best safe loss value
          - "best_params_json": JSON-serialisable dict of best-fit parameters
          - "best_trace": list of best-so-far losses after each run
          - "final_losses": list of final losses for each run
          - "chi2_reds": list of reduced chi^2 values for each run
          - "timing": aggregate timing info (total, per-block totals)
    """
    env_info = get_environment_info()

    loss = Loss(prob_model)
    safe_loss = make_safe_loss(loss)

    best_loss = np.inf
    best = None
    results = []
    best_trace = []
    final_losses = []
    chi2_reds = []  # === NEW: store per-run reduced chi^2 ===

    # Global timing accumulators
    t_total_start = now()
    total_t_sample_unconstrain = 0.0
    total_t_adam = 0.0
    total_t_lbfgs = 0.0
    total_t_io = 0.0

    for i in trange(n_starts, desc="Multi-start runs", leave=True):
        # ---- sampling from prior + unconstrain ----
        t0 = now()
        key = jax.random.PRNGKey(int(random_seed + 101 * i))  # for each run a different key
        init_params = prob_model.get_sample(prng_key=key)  # get initial params from prior
        u0 = unconstrain_fn(prob_model.model, (), {}, init_params)  # unconstrained params for JAX optimizer
        t_sample_unconstrain = now() - t0
        total_t_sample_unconstrain += t_sample_unconstrain

        # ---- initial loss + optional Adam pre-optimisation ----
        init_l = float(safe_loss(u0))
        t_adam = 0.0
        if do_preopt:
            t1 = now()
            u_start = adam_preopt(safe_loss, u0, n_steps=adam_steps, lr=adam_lr)
            t_adam = now() - t1
            total_t_adam += t_adam
            pre_l = float(safe_loss(u_start))
        else:
            u_start, pre_l = u0, init_l

        # ---- L-BFGS-B optimisation ----
        optimizer = JaxoptOptimizer(loss, loss_norm_optim=img.size)
        u_opt, logL, extra, runtime = optimizer.run_scipy(
            u_start, method="L-BFGS-B", maxiter=maxiter
        )
        t_lbfgs = float(runtime)
        total_t_lbfgs += t_lbfgs

        # ---- final loss + constraint + I/O ----
        t2 = now()
        fin_l = float(safe_loss(u_opt))
        final_losses.append(fin_l)

        params_constrained = prob_model.constrain(u_opt)  # back to constrained space

        # === NEW: compute reduced chi^2 for this run ===
        # Use prob_model.num_parameters for dof, so it's consistent with the summary.
        chi2_red = prob_model.reduced_chi2(
            params_constrained,
            n_params=prob_model.num_parameters,
        )
        chi2_reds.append(float(chi2_red))
        # === END NEW ===

        # Save best-fit parameters from this run to JSON
        params_json = {
            k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
            for k, v in params_constrained.items()
        }
        save_path = os.path.join(outdir, f"best_fit_params_run{i:02d}.json")
        with open(save_path, "w") as f:
            json.dump(params_json, f, indent=2)
        t_io = now() - t2
        total_t_io += t_io

        if verbose:
            print(
                f"[run {i:02d}] init={init_l:.3g}  pre={pre_l:.3g}  "
                f"final={fin_l:.3g}  chi2_red={chi2_red:.3g}  -> {save_path}  "
                f"(t_sample={t_sample_unconstrain:.2f}s, "
                f"t_adam={t_adam:.2f}s, t_lbfgs={t_lbfgs:.2f}s, t_io={t_io:.2f}s)"
            )

        if np.isfinite(fin_l) and fin_l < best_loss:
            best_loss = fin_l
            best = dict(run=i, u=u_opt, params=params_constrained, extra=extra)  # u_opt is unconstrained, params_constrained is constrained

        best_trace.append(best_loss)
        results.append(
            dict(
                run=i,
                init_loss=init_l,
                pre_loss=pre_l,
                final_loss=fin_l,
                chi2_red=float(chi2_red),  # === NEW: store reduced chi^2 in results ===
                runtime=float(runtime),
                t_sample_unconstrain=float(t_sample_unconstrain),
                t_adam=float(t_adam),
                t_lbfgs=float(t_lbfgs),
                t_io=float(t_io),
                is_best=(best is not None and best["run"] == i),
            )
        )

    t_total = now() - t_total_start

    summary = dict(
        n_starts=n_starts,
        n_param=prob_model.num_parameters,
        results=results,
        best_run=int(best["run"]),
        best_loss=float(best_loss),
        timing=dict(
            total=float(t_total),
            total_t_sample_unconstrain=float(total_t_sample_unconstrain),
            total_t_adam=float(total_t_adam),
            total_t_lbfgs=float(total_t_lbfgs),
            total_t_io=float(total_t_io),
        ),
        env=env_info,
    )
    # Add extra fields BEFORE saving
    summary["best_trace"] = best_trace
    summary["final_losses"] = final_losses
    summary["chi2_reds"] = chi2_reds
    summary["best_params_json"] = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
        for k, v in best["params"].items()
    }

    # Now save the COMPLETE summary
    summary_path = os.path.join(outdir, "multi_start_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"Multi-start summary saved to: {summary_path}")
        print(
            f"Timing summary: total={t_total:.2f}s, "
            f"sample+unconstrain={total_t_sample_unconstrain:.2f}s, "
            f"Adam={total_t_adam:.2f}s, "
            f"L-BFGS-B={total_t_lbfgs:.2f}s, "
            f"I/O={total_t_io:.2f}s"
        )
    return summary
def lbfgs_batch_optimize(
    safe_loss_fn,
    u_start_batch,
    maxiter: int = 600,
    tol: float = 1e-3,
):
    """
    Run jaxopt.LBFGS on a batch of initial points in parallel using vmap.

    Parameters
    ----------
    safe_loss_fn : callable
        Scalar loss function, maps a single PyTree of params -> scalar.
    u_start_batch : PyTree
        Same structure as a single params PyTree, but each leaf has a
        leading axis of size n_starts (batched initial points).
    maxiter : int
        Maximum LBFGS iterations per start.
    tol : float
        LBFGS stopping tolerance.

    Returns
    -------
    u_opt_batch : PyTree
        Batched optimized parameters; same structure as u_start_batch
        with leading axis n_starts.
    """
    solver = jaxopt.LBFGS(
        fun=safe_loss_fn,
        value_and_grad=True,
        maxiter=maxiter,
        tol=tol,
    )

    def _run_single(u0):
        # solver.run returns an OptStep(params, state)
        opt_step = solver.run(u0)
        return opt_step.params

    # Vectorise over the leading axis of the PyTree
    u_opt_batch = jax.vmap(_run_single)(u_start_batch)
    return u_opt_batch


def adam_preopt_vmap(
    safe_loss_fn,
    u0_batch,
    n_steps: int = 200,
    lr: float = 5e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
):
    """
    Batched Adam pre-optimisation of multiple unconstrained parameter vectors
    using jax.vmap, for PyTree parameters.

    Parameters
    ----------
    safe_loss_fn : callable
        Function mapping a single parameter PyTree -> scalar loss (JAX-compatible).
    u0_batch : PyTree
        Batched parameters. Same tree structure as a single u, but each leaf
        has a leading axis of size n_starts (i.e. shape (n_starts, ...)).
    n_steps : int
        Number of Adam steps.
    lr : float
        Learning rate.
    beta1, beta2, eps : float
        Standard Adam hyperparameters.

    Returns
    -------
    u_best : PyTree
        Batched best parameters found during Adam; same structure as u0_batch.
    init_loss : jnp.ndarray, shape (n_starts,)
        Loss values at the original u0_batch.
    best_loss : jnp.ndarray, shape (n_starts,)
        Best loss values found during Adam.
    """
    loss_grad = jax.grad(safe_loss_fn)

    def single_val_grad(u):
        # returns (loss, grad) for a single (unbatched) PyTree u
        return safe_loss_fn(u), loss_grad(u)

    # Vectorised over the leading axis of each leaf in the PyTree
    batched_val_grad = jax.vmap(single_val_grad)

    @jax.jit
    def _run(u_init):
        loss0, _ = batched_val_grad(u_init)  # shape (n_starts,)
        # Zero-initialise Adam moments with same PyTree structure
        m0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), u_init)
        v0 = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), u_init)
        best_u0 = u_init
        best_loss0 = loss0

        def body(step, state):
            u, m, v, best_u, best_loss = state
            loss_batch, grad_batch = batched_val_grad(u)
            t = step + 1

            # Adam updates on the whole PyTree; n_starts is just part of the shape.
            m = jax.tree_util.tree_map(
                lambda m_leaf, g_leaf: beta1 * m_leaf + (1.0 - beta1) * g_leaf,
                m, grad_batch,
            )
            v = jax.tree_util.tree_map(
                lambda v_leaf, g_leaf: beta2 * v_leaf + (1.0 - beta2) * (g_leaf * g_leaf),
                v, grad_batch,
            )

            m_hat = jax.tree_util.tree_map(
                lambda m_leaf: m_leaf / (1.0 - beta1 ** t), m
            )
            v_hat = jax.tree_util.tree_map(
                lambda v_leaf: v_leaf / (1.0 - beta2 ** t), v
            )

            u = jax.tree_util.tree_map(
                lambda u_leaf, mh, vh: u_leaf - lr * mh / (jnp.sqrt(vh) + eps),
                u, m_hat, v_hat,
            )

            # Track best per start
            better = loss_batch < best_loss  # shape (n_starts,)

            def update_best_leaf(u_leaf, best_leaf):
                # u_leaf has shape (n_starts, ...)
                # Broadcast mask over trailing dims
                shape = (better.shape[0],) + (1,) * (u_leaf.ndim - 1)
                mask = better.reshape(shape)
                return jnp.where(mask, u_leaf, best_leaf)

            best_u = jax.tree_util.tree_map(update_best_leaf, u, best_u)
            best_loss = jnp.where(better, loss_batch, best_loss)

            return (u, m, v, best_u, best_loss)

        state0 = (u_init, m0, v0, best_u0, best_loss0)
        u_fin, m_fin, v_fin, best_u, best_loss = jax.lax.fori_loop(
            0, n_steps, body, state0
        )
        return best_u, loss0, best_loss

    u_best, loss0, best_loss = _run(u0_batch)
    return u_best, loss0, best_loss

def run_multistart_vmap(
    prob_model,
    img: np.ndarray,
    noise_map: np.ndarray,
    outdir: str,
    n_starts: int = 20,
    random_seed: int = 73,
    do_preopt: bool = True,
    adam_steps: int = 250,
    adam_lr: float = 3e-3,
    maxiter: int = 600,
    rel_eps: float = 0.01,
    verbose: bool = True,
) -> Dict:
    """
    Multi-start optimisation using samples from the NumPyro prior as starting
    points, with both:
      - a batched Adam pre-optimisation (jax.vmap),
      - and a batched jaxopt.LBFGS refinement (jax.vmap),
    all in pure JAX (no SciPy).

    Parameters
    ----------
    prob_model : tdlmc_model.ProbModel
    img, noise_map : np.ndarray
        Data and per-pixel noise (for Loss / chi^2 etc.).
    outdir : str
        Directory to save per-run best-fit JSON and summary.
    n_starts : int
        Number of random starting points.
    random_seed : int
        Base RNG seed.
    do_preopt : bool
        If True, run batched Adam pre-optimisation before LBFGS.
    adam_steps : int
        Number of Adam iterations in pre-optimisation.
    adam_lr : float
        Adam learning rate.
    maxiter : int
        Maximum LBFGS iterations per start.
    rel_eps : float
        Tolerance parameter; passed to LBFGS as `tol`.
    verbose : bool
        Print per-run and timing info.

    Returns
    -------
    summary : dict
        Same structure as before, but with LBFGS done in pure JAX.
    """
    env_info = get_environment_info()
    loss = Loss(prob_model)
    safe_loss = make_safe_loss(loss)

    t_total_start = now()

    # ---- 1. Draw all starting points and unconstrain them ----
    t0 = now()
    u0_list = []
    for i in range(n_starts):
        key = jax.random.PRNGKey(int(random_seed + 101 * i))
        init_params = prob_model.get_sample(prng_key=key)
        u0 = unconstrain_fn(prob_model.model, (), {}, init_params)
        u0_list.append(u0)

    # list-of-PyTrees -> PyTree-of-batched-arrays
    u0_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs, axis=0),
        *u0_list,
    )
    t_draw_unconstrain = now() - t0

    # ---- 2. Batched Adam pre-optimisation (optional) ----
    if do_preopt:
        t1 = now()
        u_start_batch, init_l_batch, pre_l_batch = adam_preopt_vmap(
            safe_loss_fn=safe_loss,
            u0_batch=u0_batch,
            n_steps=adam_steps,
            lr=adam_lr,
        )
        t_adam_vmap = now() - t1
    else:
        batched_safe_loss = jax.jit(jax.vmap(safe_loss))
        init_l_batch = batched_safe_loss(u0_batch)
        pre_l_batch = init_l_batch
        u_start_batch = u0_batch
        t_adam_vmap = 0.0

    init_l_batch_np = np.asarray(init_l_batch)
    pre_l_batch_np = np.asarray(pre_l_batch)

    # ---- 3. Batched LBFGS refinement (pure JAX, vmap over starts) ----
    t2 = now()
    u_opt_batch = lbfgs_batch_optimize(
        safe_loss_fn=safe_loss,
        u_start_batch=u_start_batch,
        maxiter=maxiter,
        tol=rel_eps,
    )
    t_lbfgs_total = now() - t2

    # Final losses for all runs in one batched pass
    final_l_batch = jax.vmap(safe_loss)(u_opt_batch)
    final_l_batch_np = np.asarray(final_l_batch)

    # ---- 4. Per-run chi^2 + I/O, best tracking (Python loop) ----
    best_loss_val = np.inf
    best = None
    results = []
    best_trace = []
    final_losses = []
    chi2_reds = []

    total_t_io = 0.0
    # approximate per-run LBFGS runtime (just for bookkeeping)
    runtime_per_lbfgs = float(t_lbfgs_total) / float(max(n_starts, 1))

    for i in trange(n_starts, desc="Multi-start runs (vmap+LBFGS)", leave=True):
        init_l = float(init_l_batch_np[i])
        pre_l = float(pre_l_batch_np[i])
        fin_l = float(final_l_batch_np[i])
        final_losses.append(fin_l)

        # Extract i-th optimised params from batched PyTree
        u_opt_i = jax.tree_util.tree_map(lambda x: x[i], u_opt_batch)

        t_io_start = now()
        params_constrained = prob_model.constrain(u_opt_i)

        chi2_red = prob_model.reduced_chi2(
            params_constrained,
            n_params=prob_model.num_parameters,
        )
        chi2_reds.append(float(chi2_red))

        params_json = {
            k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
            for k, v in params_constrained.items()
        }
        save_path = os.path.join(outdir, f"best_fit_params_run{i:02d}.json")
        with open(save_path, "w") as f:
            json.dump(params_json, f, indent=2)
        t_io = now() - t_io_start
        total_t_io += t_io

        if verbose:
            print(
                f"[run {i:02d} (vmap+LBFGS)] init={init_l:.3g}  pre={pre_l:.3g}  "
                f"final={fin_l:.3g}  chi2_red={chi2_red:.3g}  -> {save_path}  "
                f"(t_lbfgs~{runtime_per_lbfgs:.2f}s, t_io={t_io:.2f}s)"
            )

        if np.isfinite(fin_l) and fin_l < best_loss_val:
            best_loss_val = fin_l
            best = dict(run=i, u=u_opt_i, params=params_constrained, extra=None)

        best_trace.append(best_loss_val)
        results.append(
            dict(
                run=i,
                init_loss=init_l,
                pre_loss=pre_l,
                final_loss=fin_l,
                chi2_red=float(chi2_red),
                runtime=runtime_per_lbfgs,
                t_lbfgs=runtime_per_lbfgs,
                t_io=float(t_io),
                is_best=(best is not None and best["run"] == i),
            )
        )

    t_total = now() - t_total_start

    # ---- 5. Build and save summary dict ----
    summary = dict(
        n_starts=n_starts,
        n_param=prob_model.num_parameters,
        results=results,
        best_run=int(best["run"]),
        best_loss=float(best_loss_val),
        timing=dict(
            total=float(t_total),
            t_draw_unconstrain=float(t_draw_unconstrain),
            t_adam_vmap=float(t_adam_vmap),
            total_t_lbfgs=float(t_lbfgs_total),
            total_t_io=float(total_t_io),
        ),
        env=env_info,
    )

    summary["best_trace"] = best_trace
    summary["final_losses"] = final_losses
    summary["chi2_reds"] = chi2_reds
    summary["best_params_json"] = {
        k: (float(v) if np.ndim(v) == 0 else np.asarray(v).tolist())
        for k, v in best["params"].items()
    }

    summary_path = os.path.join(outdir, "multi_start_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"Multi-start (vmap+LBFGS) summary saved to: {summary_path}")
        print(
            f"Timing summary (vmap+LBFGS): total={t_total:.2f}s, "
            f"draw+unconstrain={t_draw_unconstrain:.2f}s, "
            f"Adam(vmap)={t_adam_vmap:.2f}s, "
            f"LBFGS(vmap)={t_lbfgs_total:.2f}s, "
            f"I/O={total_t_io:.2f}s"
        )

    return summary

def tnorm(mu, sigma, lo, hi):
    """Convenience wrapper around scipy.stats.truncnorm using (mu, sigma, lo, hi)."""
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return truncnorm(a, b, loc=mu, scale=sigma)


def build_nautilus_prior(best_params: Dict) -> Tuple[Prior, Dict, int]:
    """
    Build a NAUTILUS Prior object around a given best-fit parameter dict.

    Parameters
    ----------
    best_params : dict
        Dictionary as loaded from a `best_fit_params_runXX.json` file produced
        by `run_multistart` (i.e. with keys like 'lens_theta_E', 'x_image', ...).

    Returns
    -------
    prior : nautilus.Prior
    best_flat : dict
        Copy of best_params where the point-source arrays are flattened into
        x_image_0, y_image_0, ps_amp_0, ...
    nps : int
        Number of point-source images.
    """
    best_flat = dict(best_params)
    x_im = np.atleast_1d(best_params.get("x_image", []))
    y_im = np.atleast_1d(best_params.get("y_image", []))
    amp_im = np.atleast_1d(best_params.get("ps_amp", []))
    nps = len(x_im)

    for i in range(nps):
        best_flat[f"x_image_{i}"] = float(x_im[i])
        best_flat[f"y_image_{i}"] = float(y_im[i])
        best_flat[f"ps_amp_{i}"] = float(amp_im[i])

    prior = Prior()

    # --- Mass + shear ---
    prior.add_parameter("lens_center_x", dist=norm(loc=best_flat["lens_center_x"], scale=0.3))
    prior.add_parameter("lens_center_y", dist=norm(loc=best_flat["lens_center_y"], scale=0.3))
    prior.add_parameter("lens_theta_E", dist=tnorm(best_flat["lens_theta_E"], 0.3, 0.3, 2.2))
    prior.add_parameter("lens_e1", dist=tnorm(best_flat["lens_e1"], 0.2, -0.4, 0.4))
    prior.add_parameter("lens_e2", dist=tnorm(best_flat["lens_e2"], 0.2, -0.4, 0.4))
    prior.add_parameter("lens_gamma", dist=tnorm(best_flat["lens_gamma"], 0.25, 1.2, 2.8))
    prior.add_parameter("lens_gamma1", dist=tnorm(best_flat["lens_gamma1"], 0.15, -0.3, 0.3))
    prior.add_parameter("lens_gamma2", dist=tnorm(best_flat["lens_gamma2"], 0.15, -0.3, 0.3))

    # --- Lens light ---
    mu_Lamp = np.log(max(best_flat["light_amp_L"], 1e-8))
    prior.add_parameter("light_amp_L", dist=lognorm(s=1.0, scale=np.exp(mu_Lamp)))
    prior.add_parameter("light_Re_L", dist=tnorm(best_flat["light_Re_L"], 0.25, 0.05, 2.5))
    prior.add_parameter("light_n_L", dist=tnorm(best_flat["light_n_L"], 0.5, 0.7, 5.5))
    prior.add_parameter("light_e1_L", dist=tnorm(best_flat["light_e1_L"], 0.2, -0.6, 0.6))
    prior.add_parameter("light_e2_L", dist=tnorm(best_flat["light_e2_L"], 0.2, -0.6, 0.6))

    # --- Source light ---
    mu_Samp = np.log(max(best_flat["light_amp_S"], 1e-8))
    prior.add_parameter("light_amp_S", dist=lognorm(s=1.2, scale=np.exp(mu_Samp)))
    prior.add_parameter("light_Re_S", dist=tnorm(best_flat["light_Re_S"], 0.2, 0.03, 1.2))
    prior.add_parameter("light_n_S", dist=tnorm(best_flat["light_n_S"], 0.5, 0.5, 4.5))
    prior.add_parameter("src_center_x", dist=norm(loc=best_flat["src_center_x"], scale=0.6))
    prior.add_parameter("src_center_y", dist=norm(loc=best_flat["src_center_y"], scale=0.6))
    prior.add_parameter(
        "light_e1_S",
        dist=tnorm(best_flat.get("light_e1_S", 0.0), 0.35, -0.8, 0.8),
    )
    prior.add_parameter(
        "light_e2_S",
        dist=tnorm(best_flat.get("light_e2_S", 0.0), 0.35, -0.8, 0.8),
    )

    # --- Point sources ---
    for i in range(nps):
        prior.add_parameter(
            f"x_image_{i}",
            dist=norm(loc=best_flat[f"x_image_{i}"], scale=0.2),
        )
        prior.add_parameter(
            f"y_image_{i}",
            dist=norm(loc=best_flat[f"y_image_{i}"], scale=0.2),
        )
        mu_A = np.log(max(best_flat[f"ps_amp_{i}"], 1e-10))
        prior.add_parameter(
            f"ps_amp_{i}",
            dist=lognorm(s=0.6, scale=np.exp(mu_A)),
        )

    return prior, best_flat, nps


def make_paramdict_to_kwargs(best_flat: Dict, nps: int):
    """
    Build a closure that maps a NAUTILUS sample dict into kwargs for LensImage.model.
    Any parameter not present in the sample dict is taken from `best_flat`.
    """
    def paramdict_to_kwargs(d: Dict):
        P = dict(best_flat)
        P.update({k: d[k] for k in d.keys() if k in P})

        kwargs_lens = [
            dict(
                theta_E=P["lens_theta_E"],
                e1=P["lens_e1"],
                e2=P["lens_e2"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
                gamma=P["lens_gamma"],
            ),
            dict(
                gamma1=P["lens_gamma1"],
                gamma2=P["lens_gamma2"],
                ra_0=0.0,
                dec_0=0.0,
            ),
        ]
        kwargs_lens_light = [
            dict(
                amp=P["light_amp_L"],
                R_sersic=P["light_Re_L"],
                n_sersic=P["light_n_L"],
                e1=P["light_e1_L"],
                e2=P["light_e2_L"],
                center_x=P["lens_center_x"],
                center_y=P["lens_center_y"],
            )
        ]
        kwargs_source = [
            dict(
                amp=P["light_amp_S"],
                R_sersic=P["light_Re_S"],
                n_sersic=P["light_n_S"],
                e1=P["light_e1_S"],
                e2=P["light_e2_S"],
                center_x=P["src_center_x"],
                center_y=P["src_center_y"],
            )
        ]
        if nps:
            ra = np.array([P[f"x_image_{i}"] for i in range(nps)])
            dec = np.array([P[f"y_image_{i}"] for i in range(nps)])
            amp = np.array([P[f"ps_amp_{i}"] for i in range(nps)])
            kwargs_point = [dict(ra=ra, dec=dec, amp=amp)]
        else:
            kwargs_point = None

        return dict(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

    return paramdict_to_kwargs


def build_gaussian_loglike(lens_image, img: np.ndarray, noise_map: np.ndarray, paramdict_to_kwargs):
    """
    Build a Gaussian pixel log-likelihood closure suitable for NAUTILUS.

    logL = -0.5 * sum( (r/σ)^2 + log(2πσ^2) )
    where r = data - model.
    """
    good = np.isfinite(noise_map) & (noise_map > 0)
    sigma2 = (noise_map[good] ** 2)
    const_term = np.log(2.0 * np.pi * sigma2)

    def loglike(sample_dict: Dict):
        kw = paramdict_to_kwargs(sample_dict)
        model = lens_image.model(**kw)
        r = (img - model)[good]
        return float(-0.5 * np.sum(r * r / sigma2 + const_term))

    return loglike


def build_nautilus_prior_and_loglike(best_params: Dict, lens_image, img: np.ndarray, noise_map: np.ndarray):
    """
    Convenience function that, given a best-fit parameter dict and the
    lens_image/data, returns (prior, paramdict_to_kwargs, loglike) ready for NAUTILUS.
    """
    prior, best_flat, nps = build_nautilus_prior(best_params)
    paramdict_to_kwargs = make_paramdict_to_kwargs(best_flat, nps)
    loglike = build_gaussian_loglike(lens_image, img, noise_map, paramdict_to_kwargs)
    return prior, paramdict_to_kwargs, loglike


# ----------------------------------------------------------------------
# EMC / emcee ENSEMBLE MCMC HELPERS
# ----------------------------------------------------------------------


def build_emcee_prior(best_params: Dict):
    """
    Build an ordered list of parameter names and corresponding SciPy
    prior distributions for use with an 'emcee' ensemble MCMC sampler.

    The parametrisation and widths are identical to build_nautilus_prior,
    so EMC and NAUTILUS explore the *same* prior volume.

    Parameters
    ----------
    best_params : dict
        Dictionary as loaded from best_fit_params_runXX.json.

    Returns
    -------
    param_names : list of str
        Names of the parameters in the order expected by theta vectors.
    prior_dists : list of scipy.stats distributions
        One distribution per parameter.
    best_flat : dict
        Flattened version of best_params (x_image_0, y_image_0, ...).
    nps : int
        Number of point-source images.
    """
    # Flatten point-source arrays exactly as in build_nautilus_prior
    best_flat = dict(best_params)
    x_im = np.atleast_1d(best_params.get("x_image", []))
    y_im = np.atleast_1d(best_params.get("y_image", []))
    amp_im = np.atleast_1d(best_params.get("ps_amp", []))
    nps = len(x_im)

    for i in range(nps):
        best_flat[f"x_image_{i}"] = float(x_im[i])
        best_flat[f"y_image_{i}"] = float(y_im[i])
        best_flat[f"ps_amp_{i}"] = float(amp_im[i])

    param_names = []
    prior_dists = []

    def add_param(name, dist):
        param_names.append(name)
        prior_dists.append(dist)

    # --- Mass + shear ---
    add_param("lens_center_x", norm(loc=best_flat["lens_center_x"], scale=0.3))
    add_param("lens_center_y", norm(loc=best_flat["lens_center_y"], scale=0.3))
    add_param("lens_theta_E", tnorm(best_flat["lens_theta_E"], 0.3, 0.3, 2.2))
    add_param("lens_e1", tnorm(best_flat["lens_e1"], 0.2, -0.4, 0.4))
    add_param("lens_e2", tnorm(best_flat["lens_e2"], 0.2, -0.4, 0.4))
    add_param("lens_gamma", tnorm(best_flat["lens_gamma"], 0.25, 1.2, 2.8))
    add_param("lens_gamma1", tnorm(best_flat["lens_gamma1"], 0.15, -0.3, 0.3))
    add_param("lens_gamma2", tnorm(best_flat["lens_gamma2"], 0.15, -0.3, 0.3))

    # --- Lens light ---
    mu_Lamp = np.log(max(best_flat["light_amp_L"], 1e-8))
    add_param("light_amp_L", lognorm(s=1.0, scale=np.exp(mu_Lamp)))
    add_param("light_Re_L", tnorm(best_flat["light_Re_L"], 0.25, 0.05, 2.5))
    add_param("light_n_L", tnorm(best_flat["light_n_L"], 0.5, 0.7, 5.5))
    add_param("light_e1_L", tnorm(best_flat["light_e1_L"], 0.2, -0.6, 0.6))
    add_param("light_e2_L", tnorm(best_flat["light_e2_L"], 0.2, -0.6, 0.6))

    # --- Source light ---
    mu_Samp = np.log(max(best_flat["light_amp_S"], 1e-8))
    add_param("light_amp_S", lognorm(s=1.2, scale=np.exp(mu_Samp)))
    add_param("light_Re_S", tnorm(best_flat["light_Re_S"], 0.2, 0.03, 1.2))
    add_param("light_n_S", tnorm(best_flat["light_n_S"], 0.5, 0.5, 4.5))
    add_param("src_center_x", norm(loc=best_flat["src_center_x"], scale=0.6))
    add_param("src_center_y", norm(loc=best_flat["src_center_y"], scale=0.6))
    add_param(
        "light_e1_S",
        tnorm(best_flat.get("light_e1_S", 0.0), 0.35, -0.8, 0.8),
    )
    add_param(
        "light_e2_S",
        tnorm(best_flat.get("light_e2_S", 0.0), 0.35, -0.8, 0.8),
    )

    # --- Point sources ---
    for i in range(nps):
        add_param(
            f"x_image_{i}",
            norm(loc=best_flat[f"x_image_{i}"], scale=0.2),
        )
        add_param(
            f"y_image_{i}",
            norm(loc=best_flat[f"y_image_{i}"], scale=0.2),
        )
        mu_A = np.log(max(best_flat[f"ps_amp_{i}"], 1e-10))
        add_param(
            f"ps_amp_{i}",
            lognorm(s=0.6, scale=np.exp(mu_A)),
        )

    return param_names, prior_dists, best_flat, nps


def _vector_to_paramdict(theta: np.ndarray, param_names):
    """
    Convert a 1D parameter vector into a dict mapping param_names -> value.
    """
    return {name: float(val) for name, val in zip(param_names, theta)}


def build_emcee_logprob(
    param_names,
    prior_dists,
    best_flat: Dict,
    nps: int,
    lens_image,
    img: np.ndarray,
    noise_map: np.ndarray,
):
    """
    Build an 'emcee'-compatible log-probability function:

        log_prob(theta) = log_prior(theta) + log_like(theta)

    where theta is a 1D numpy array ordered according to param_names.

    Parameters
    ----------
    param_names : list[str]
        Names of parameters in the same order as theta.
    prior_dists : list[scipy.stats]
        Prior distributions for each parameter.
    best_flat, nps :
        As returned by build_emcee_prior; used to construct kwargs dicts.
    lens_image, img, noise_map :
        Same objects as for build_gaussian_loglike.

    Returns
    -------
    log_prob : callable
        Function mapping theta -> log posterior.
    paramdict_to_kwargs : callable
        Same as make_paramdict_to_kwargs(best_flat, nps).
    loglike : callable
        Underlying Gaussian pixel log-likelihood (dict -> float).
    """
    paramdict_to_kwargs = make_paramdict_to_kwargs(best_flat, nps)
    loglike = build_gaussian_loglike(lens_image, img, noise_map, paramdict_to_kwargs)

    def log_prior(theta):
        theta = np.asarray(theta)
        if not np.all(np.isfinite(theta)):
            return -np.inf
        lp = 0.0
        for val, dist in zip(theta, prior_dists):
            lpi = dist.logpdf(val)
            if not np.isfinite(lpi):
                return -np.inf
            lp += float(lpi)
        return lp

    def log_prob(theta):
        # Prior
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        # Build sample dict & call Gaussian loglike
        sample_dict = _vector_to_paramdict(theta, param_names)
        try:
            ll = loglike(sample_dict)
        except Exception:
            # If the forward model explodes, just reject this point.
            return -np.inf

        if not np.isfinite(ll):
            return -np.inf

        return lp + float(ll)

    return log_prob, paramdict_to_kwargs, loglike


def draw_emcee_initial_positions(
    prior_dists,
    n_walkers: int = 32,
    random_seed: int = 73,
):
    """
    Draw initial walker positions for emcee ensemble MCMC **from the prior**.

    Parameters
    ----------
    prior_dists : list[scipy.stats]
        One distribution per parameter.
    n_walkers : int
        Number of walkers in the ensemble.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    p0 : np.ndarray, shape (n_walkers, ndim)
        Initial positions for all walkers.
    """
    rs = np.random.RandomState(random_seed)
    ndim = len(prior_dists)
    p0 = np.empty((n_walkers, ndim), dtype=float)

    for j, dist in enumerate(prior_dists):
        # SciPy distributions support random_state in recent versions.
        # If not available, this will just ignore the kwarg.
        try:
            p0[:, j] = dist.rvs(size=n_walkers, random_state=rs)
        except TypeError:
            np.random.set_state(rs.get_state())
            p0[:, j] = dist.rvs(size=n_walkers)

    return p0

class TimedLogProb:
    """
    Wrapper around an emcee log-probability function that keeps track of
    how many times it is called and how much wall time it consumes.
    """

    def __init__(self, log_prob):
        self.log_prob = log_prob
        self.n_calls = 0
        self.total_time = 0.0
        self.max_time = 0.0

    def __call__(self, theta):
        t0 = now()
        val = self.log_prob(theta)
        dt = now() - t0
        self.n_calls += 1
        self.total_time += dt
        if dt > self.max_time:
            self.max_time = dt
        return val

def run_emcee(
    log_prob,
    p0: np.ndarray,
    n_steps: int = 5000,
    backend_path: str = None,
    progress: bool = True,
):
    """
    Run EMC / emcee ensemble MCMC given a log-prob function and initial positions.

    Parameters
    ----------
    log_prob : callable
        Function mapping theta -> log posterior.
    p0 : np.ndarray, shape (n_walkers, ndim)
        Initial positions of the walkers.
    n_steps : int
        Number of MCMC steps to run.
    backend_path : str or None
        If not None, use an emcee HDFBackend at this path (reset each run).
        This gives you disk-backed chains you can inspect later.
    progress : bool
        If True, show emcee's progress bar.

    Returns
    -------
    sampler : emcee.EnsembleSampler
        The sampler instance (contains chain, log_prob, etc.).
    chain : np.ndarray
        The MCMC chain with shape (n_steps, n_walkers, ndim).
    log_prob_chain : np.ndarray
        Log-posterior values with shape (n_steps, n_walkers).
    summary : dict
        Detailed run information:
            - "n_walkers"
            - "n_steps"
            - "ndim"
            - "runtime"
            - "mean_acceptance_fraction"
            - "n_log_prob_calls"
            - "log_prob_total"
            - "log_prob_max"
            - "log_prob_avg"
            - "sampler_overhead"
            - "n_samples"
            - "samples_per_second"
            - "effective_samples_per_second"
            - "autocorr_time" (if available)
            - "mean_autocorr_time" (if available)
            - "env" (environment snapshot)
    """
    p0 = np.asarray(p0, dtype=float)
    n_walkers, ndim = p0.shape

    # Wrap log_prob to time calls
    timed_log_prob = TimedLogProb(log_prob)

    if backend_path is not None:
        backend_dir = os.path.dirname(os.path.abspath(backend_path))
        if backend_dir and not os.path.isdir(backend_dir):
            os.makedirs(backend_dir, exist_ok=True)
        backend = emcee.backends.HDFBackend(backend_path)
        # Reset so we don't accidentally append to an old chain.
        backend.reset(n_walkers, ndim)
    else:
        backend = None

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        timed_log_prob,
        backend=backend,
    )

    t0 = now()
    sampler.run_mcmc(p0, n_steps, progress=progress)
    t1 = now()

    runtime = float(t1 - t0)
    chain = sampler.get_chain()          # (n_steps, n_walkers, ndim)
    log_prob_chain = sampler.get_log_prob()  # (n_steps, n_walkers)

    # Basic stats
    mean_af = float(np.mean(sampler.acceptance_fraction))
    n_samples = int(n_steps * n_walkers)

    # Timing stats from the wrapper
    n_calls = int(timed_log_prob.n_calls)
    log_tot = float(timed_log_prob.total_time)
    log_max = float(timed_log_prob.max_time)
    if n_calls > 0:
        log_avg = log_tot / n_calls
    else:
        log_avg = 0.0

    overhead = runtime - log_tot
    if runtime > 0.0:
        samples_per_second = n_samples / runtime
    else:
        samples_per_second = 0.0
    effective_samples_per_second = samples_per_second * mean_af

    # Try to estimate integrated autocorrelation times (might fail for short runs)
    try:
        tau = sampler.get_autocorr_time(quiet=True)
        tau_arr = np.asarray(tau, dtype=float)
        mean_tau = float(np.mean(tau_arr))
        tau_list = tau_arr.tolist()
    except Exception:
        tau_list = None
        mean_tau = None

    summary = dict(
        n_walkers=int(n_walkers),
        n_steps=int(n_steps),
        ndim=int(ndim),
        runtime=runtime,
        mean_acceptance_fraction=mean_af,
        n_log_prob_calls=n_calls,
        log_prob_total=log_tot,
        log_prob_max=log_max,
        log_prob_avg=log_avg,
        sampler_overhead=float(overhead),
        n_samples=n_samples,
        samples_per_second=float(samples_per_second),
        effective_samples_per_second=float(effective_samples_per_second),
        autocorr_time=tau_list,
        mean_autocorr_time=mean_tau,
        env=get_environment_info(),
    )

    # Also attach a convenient attribute to the sampler itself
    sampler.benchmark = summary

    return sampler, chain, log_prob_chain, summary


# ----------------------------------------------------------------------
# NAUTILUS timing wrapper
# ----------------------------------------------------------------------

class TimedLoglike:
    """
    Wrapper around a log-likelihood function that keeps track of how many times
    it is called and how much wall time it consumes.
    """

    def __init__(self, loglike):
        self.loglike = loglike
        self.n_calls = 0
        self.total_time = 0.0
        self.max_time = 0.0

    def __call__(self, sample_dict: Dict):
        t0 = now()
        val = self.loglike(sample_dict)
        dt = now() - t0
        self.n_calls += 1
        self.total_time += dt
        if dt > self.max_time:
            self.max_time = dt
        return val


def _save_nautilus_timing_json(timing_dict: Dict, filepath: str) -> str:
    """
    Save NAUTILUS timing information next to the checkpoint, under:

        <dir_of_filepath>/logs/benchmark_timing/timing_<ckptname>_<timestamp>.json

    Returns the full path to the written JSON file.
    """
    base_dir = os.path.dirname(os.path.abspath(filepath))
    log_dir = os.path.join(base_dir, "logs", "benchmark_timing")
    os.makedirs(log_dir, exist_ok=True)

    ckpt_name = os.path.splitext(os.path.basename(filepath))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"timing_{ckpt_name}_{timestamp}.json"
    out_path = os.path.join(log_dir, out_name)

    payload = {
        "checkpoint": os.path.abspath(filepath),
        "timestamp": timestamp,
        "timing": timing_dict,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    return out_path


def _attach_timing_from_json_if_available(sampler: Sampler, filepath: str) -> None:
    """
    Try to load the most recent timing JSON corresponding to `filepath`
    and attach it as `sampler.timing`. Does nothing if no log is found.
    """
    base_dir = os.path.dirname(os.path.abspath(filepath))
    log_dir = os.path.join(base_dir, "logs", "benchmark_timing")
    if not os.path.isdir(log_dir):
        return

    ckpt_name = os.path.splitext(os.path.basename(filepath))[0]
    prefix = f"timing_{ckpt_name}_"
    candidates = [
        fname for fname in os.listdir(log_dir)
        if fname.startswith(prefix) and fname.endswith(".json")
    ]
    if not candidates:
        return

    # Filenames contain a YYYYMMDD_HHMMSS timestamp -> lexicographically sortable
    latest = max(candidates)
    timing_path = os.path.join(log_dir, latest)

    try:
        with open(timing_path, "r") as f:
            payload = json.load(f)
        timing = payload.get("timing", payload)
        sampler.timing = timing
    except Exception:
        # If anything goes wrong, just skip silently – we don't want to break loading.
        pass

def run_nautilus(
    prior: Prior,
    loglike,
    n_live: int,
    filepath: str,
    resume: bool = False,
    verbose: bool = True,
    run_kwargs: Dict = None,
):
    """
    Run NAUTILUS nested sampling and return (sampler, points, log_w, log_l).

    Results are saved in `filepath` (HDF5) so you can resume later.

    Additionally, timing information is attached to the returned sampler as
    `sampler.timing`, containing:
      - "total_runtime"
      - "n_loglike_calls"
      - "loglike_total"
      - "loglike_max"
      - "avg_loglike"
      - "sampler_overhead"
      - "calls_per_second"
      - "n_live"
      - "n_dim"
      - plus environment info

    Parameters
    ----------
    prior : nautilus.Prior
    loglike : callable
        Dict -> float, Gaussian pixel log-likelihood.
    n_live : int
        Number of live points (kept fixed; do NOT reduce for benchmarking).
    filepath : str
        HDF5 checkpoint path.
    resume : bool
        If True, resume from existing checkpoint.
    verbose : bool
        Convenience flag; used if `verbose` is not specified in `run_kwargs`.
    run_kwargs : dict or None
        Extra keyword arguments forwarded to `Sampler.run`. Typical ones are
        those from the NAUTILUS API:

            run(
                f_live=0.01,
                n_shell=1,
                n_eff=10000,
                n_like_max=np.inf,
                discard_exploration=False,
                timeout=np.inf,
                verbose=False,
            )

        For a *benchmark* run you would, for example, keep `n_live` the same
        but choose a smaller `n_eff` and/or a finite `n_like_max` and/or
        `timeout`.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    # Force-disable HDF5 file locking for this process
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    timed_loglike = TimedLoglike(loglike)

    if run_kwargs is None:
        run_kwargs = {}
    else:
        run_kwargs = dict(run_kwargs)  # make a shallow copy

    # If the caller didn't specify verbose in run_kwargs, use our flag
    run_kwargs.setdefault("verbose", verbose)

    sampler = Sampler(
        prior,
        timed_loglike,
        n_live=n_live,
        filepath=filepath,
        resume=resume,
    )

    t0 = now()
    try:
        success = sampler.run(**run_kwargs)
        t1 = now()
        interrupted = False
    except KeyboardInterrupt:
        t1 = now()
        interrupted = True
        success = False
        if verbose:
            print("[KeyboardInterrupt] NAUTILUS sampling interrupted by user.")
            print(f"Checkpoint should be available at: {filepath}")

    total_runtime = t1 - t0
    overhead = total_runtime - timed_loglike.total_time

    if timed_loglike.n_calls > 0:
        avg_call = timed_loglike.total_time / timed_loglike.n_calls
    else:
        avg_call = 0.0

    # Attach timing info to sampler for later inspection
    sampler.timing = dict(
        total_runtime=float(total_runtime),
        n_loglike_calls=int(timed_loglike.n_calls),
        loglike_total=float(timed_loglike.total_time),
        loglike_max=float(timed_loglike.max_time),
        avg_loglike=float(avg_call),
        sampler_overhead=float(overhead),
        calls_per_second=(
            float(timed_loglike.n_calls / total_runtime)
            if total_runtime > 0 else 0.0
        ),
        success=bool(success),
    )

    # Include sampler configuration if available
    try:
        sampler.timing["n_live"] = int(getattr(sampler, "n_live"))
    except Exception:
        sampler.timing.setdefault("n_live", int(n_live))
    try:
        sampler.timing["n_dim"] = int(getattr(sampler, "n_dim"))
    except Exception:
        pass

    # Attach environment snapshot
    sampler.timing.update(get_environment_info())

    # Persist timing information to disk next to the checkpoint
    log_path = _save_nautilus_timing_json(sampler.timing, filepath)

    # Get posterior from whatever is in the checkpoint / sampler state
    points, log_w, log_l = sampler.posterior()

    if verbose:
        if interrupted:
            print(
                f"NAUTILUS Sampling interrupted. "
                f"Elapsed time: {total_runtime / 60:.2f} minutes ({total_runtime:.2f} seconds)"
            )
        else:
            print(
                f"NAUTILUS Sampling complete! "
                f"Runtime: {total_runtime / 60:.2f} minutes ({total_runtime:.2f} seconds)"
            )

        print(
            f"  loglike: {timed_loglike.n_calls} calls, "
            f"total {timed_loglike.total_time:.2f} s, "
            f"avg {avg_call:.4f} s/call, "
            f"max {timed_loglike.max_time:.4f} s"
        )
        print(f"  sampler overhead (proposals + I/O etc.): {overhead:.2f} s")
        print(f"  timing benchmarks saved to: {log_path}")

    return sampler, points, log_w, log_l


def load_posterior_from_checkpoint(
    prior: Prior,
    loglike,
    n_live: int,
    filepath: str,
):
    """
    Load an existing NAUTILUS posterior from a checkpoint file, without
    running any additional sampling.
    """
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    sampler = Sampler(prior, loglike, n_live=n_live, filepath=filepath, resume=True)
    points, log_w, log_l = sampler.posterior()
    return sampler, points, log_w, log_l


def load_multistart_summary(outdir: str, verbose: bool = True):
    """
    Load an existing multi-start summary and best-fit parameters from disk.

    Looks for:
      - multi_start_summary.json
      - best_fit_params_runXX.json   (for the best run index)

    Returns
    -------
    summary : dict
        The loaded summary with 'best_params_json' filled in.
    """
    import json as _json
    import os as _os

    summary_path = _os.path.join(outdir, "multi_start_summary.json")
    if not _os.path.exists(summary_path):
        raise FileNotFoundError(f"No multi-start summary found at {summary_path}")

    with open(summary_path, "r") as f:
        summary = _json.load(f)

    best_run = summary.get("best_run", 0)
    best_params_path = _os.path.join(outdir, f"best_fit_params_run{best_run:02d}.json")

    if not _os.path.exists(best_params_path):
        raise FileNotFoundError(f"No best-fit file found at {best_params_path}")

    with open(best_params_path, "r") as f:
        best_params = _json.load(f)

    summary["best_params_json"] = best_params
    if verbose:
        print(f"Loaded summary from: {summary_path}")
        print(f"Loaded best-fit parameters from: {best_params_path}")
    return summary


# ----------------------------------------------------------------------
# SVI HELPER-FUNKTIONEN
# ----------------------------------------------------------------------


def build_svi(
    prob_model,
    init_params: Dict,
    init_from_prior_median: bool = False,
    prior_median_num_samples: int = 25,
    learning_rate_init: float = 0.01,
    decay_rate: float = 0.99,
    transition_steps: int = 50,
):
    """
    Baue eine SVI-Instanz mit AutoLowRankMultivariateNormal-Guide für prob_model.

    Parameters
    ----------
    prob_model : tdlmc_model.ProbModel
        Probabilistisches Modell (mit .model für NumPyro).
    init_params : dict
        Startwerte im konstrainierten Parameterraum (typisch: best_params_json).
        Nur genutzt, falls init_from_prior_median == False.
    init_from_prior_median : bool
        Wenn True: init_to_median über dem Prior.
        Wenn False: init_to_value mit init_params.
    prior_median_num_samples : int
        Anzahl Samples für init_to_median.
    learning_rate_init : float
        Start-Lernrate für den Scheduler.
    decay_rate : float
        Decay-Faktor für exponential_decay.
    transition_steps : int
        Übergangssteps für exponential_decay.

    Returns
    -------
    svi : infer.SVI
    guide_svi : autoguide.AutoLowRankMultivariateNormal
    """
    model_svi = prob_model.model

    if init_from_prior_median:
        init_fun_svi = infer.init_to_median(num_samples=prior_median_num_samples)
    else:
        init_fun_svi = infer.init_to_value(values=init_params)

    guide_svi = autoguide.AutoLowRankMultivariateNormal(
        model_svi,
        init_loc_fn=init_fun_svi,
    )

    scheduler = optax.exponential_decay(
        init_value=learning_rate_init,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
    )

    optim_svi = optax.adabelief(learning_rate=scheduler)
    loss_svi = infer.TraceMeanField_ELBO(num_particles=1)

    svi = infer.SVI(
        model_svi,
        guide_svi,
        optim_svi,
        loss_svi,
    )
    return svi, guide_svi


def run_svi_chains(
    prob_model,
    init_params: Dict,
    prng_key,
    num_chains: int = 4,
    max_iterations_svi: int = 2000,
    init_from_prior_median: bool = False,
    prior_median_num_samples: int = 25,
    learning_rate_init: float = 0.01,
    decay_rate: float = 0.99,
    transition_steps: int = 50,
    progress_bar: bool = False,
    stable_update: bool = False,
    verbose: bool = True,
):
    """
    Führe mehrere SVI-Chains (mit AutoLowRankMultivariateNormal) für dasselbe
    ProbModel aus – analog zu deinem Beispiel-Snippet.

    Parameters
    ----------
    prob_model : tdlmc_model.ProbModel
        Probabilistisches Modell.
    init_params : dict
        Startwerte im konstrainierten Parameterraum (z.B. best_params_json).
    prng_key : jax.random.PRNGKey
        Basis-RNG-Key für alle Chains.
    num_chains : int
        Anzahl paralleler SVI-Chains (unabhängige Seeds).
    max_iterations_svi : int
        Anzahl SVI-Iterationen pro Chain.
    init_from_prior_median : bool
        Siehe build_svi.
    prior_median_num_samples : int
        Siehe build_svi.
    learning_rate_init, decay_rate, transition_steps
        Hyperparameter des optax.exponential_decay-Schedulers.
    progress_bar : bool
        Ob NumPyro den Fortschrittsbalken anzeigen soll.
    stable_update : bool
        Wird direkt an svi.run weitergereicht.
    verbose : bool
        Wenn True, werden einfache Statusmeldungen ausgegeben.

    Returns
    -------
    summary : dict
        Zusammenfassung mit:
          - "num_chains"
          - "max_iterations_svi"
          - "elbo_losses" (Array [num_chains, max_iterations_svi])
          - "final_elbo_losses" (Array [num_chains])
          - "median_losses" (Loss im herculens-Sinn bei den Guide-Medianen)
    svi : infer.SVI
    guide_svi : autoguide.AutoLowRankMultivariateNormal
    params_list : list[dict]
        Guide-Parameter pro Chain (für sample_posterior).
    median_list : list[dict]
        Median-Parameter (konstrainiert) pro Chain.
    median_unconstrained_list : list[jnp.ndarray]
        Entsprechende unkonstrahierte Parametervektoren pro Chain.
    """
    # SVI + Guide aufsetzen
    svi, guide_svi = build_svi(
        prob_model,
        init_params=init_params,
        init_from_prior_median=init_from_prior_median,
        prior_median_num_samples=prior_median_num_samples,
        learning_rate_init=learning_rate_init,
        decay_rate=decay_rate,
        transition_steps=transition_steps,
    )

    # RNG-Keys für die Chains
    prng_key, prng_key_svi = jax.random.split(prng_key)
    svi_keys = jax.random.split(prng_key_svi, num_chains)

    results = []
    for i in range(num_chains):
        if verbose:
            print(f"[SVI] Starte Chain {i+1}/{num_chains} mit {max_iterations_svi} Iterationen")
        res = svi.run(
            svi_keys[i],
            max_iterations_svi,
            progress_bar=progress_bar,
            stable_update=stable_update,
        )
        results.append(res)

    # Verlaufsdaten der ELBO-Losses (negative ELBO)
    losses = jnp.stack([r.losses for r in results], axis=0)  # (num_chains, max_iterations)
    final_losses = np.asarray(losses[:, -1])

    # Guide-Parameter pro Chain
    params_list = [svi.get_params(r.state) for r in results]

    # Mediane im konstrainierten Raum
    median_list = [guide_svi.median(p) for p in params_list]

    # Unkonstrahierte Vektoren zu den Medianen
    median_unconstrained_list = [
        unconstrain_fn(prob_model.model, (), {}, m)
        for m in median_list
    ]

    # Optional: Loss im herculens-Sinn an den SVI-Medianen
    loss_obj = Loss(prob_model)
    safe_loss = make_safe_loss(loss_obj)
    median_losses = [
        float(safe_loss(u))
        for u in median_unconstrained_list
    ]

    if verbose:
        for i, lval in enumerate(median_losses):
            print(f"[SVI] herculens-Loss am Median von Chain {i}: {lval:.3g}")

    summary = dict(
        num_chains=int(num_chains),
        max_iterations_svi=int(max_iterations_svi),
        elbo_losses=np.asarray(losses),
        final_elbo_losses=final_losses,
        median_losses=np.asarray(median_losses),
    )

    return summary, svi, guide_svi, params_list, median_list, median_unconstrained_list



def get_nautilus_posterior(
    sampler,
    points,
    log_w,
    log_l,
    n_samples=None,
    random_seed=None,
    use_weights=True,
):
    """
    Convert a NAUTILUS result into a standard posterior dictionary.

    Parameters
    ----------
    sampler : nautilus.Sampler
        Sampler instance returned by `run_nautilus` or `load_posterior_from_checkpoint`.
        Used only for meta info (evidence, timing).
    points :
        Points returned by `sampler.posterior()`.
        Typically a structured array with one field per parameter.
    log_w : array-like
        Log-weights returned by `sampler.posterior()`.
    log_l : array-like
        Log-likelihood values returned by `sampler.posterior()`.
    n_samples : int or None
        If None: keep all posterior points.
        If int: draw approximately `n_samples` *unweighted* samples by
        importance resampling (using the weights).
    random_seed : int or None
        Seed for the resampling RNG.
    use_weights : bool
        If True (default) and n_samples is not None, use the NAUTILUS weights
        as sampling probabilities. If False, subsample uniformly from the
        available points.

    Returns
    -------
    posterior : dict
        Standardised posterior dictionary with keys:
          - "engine" = "nautilus"
          - "samples" : (n_samples, ndim)
          - "log_likelihood" : (n_samples,)
          - "log_weights" : (n_samples,)
          - "param_names" : list[str] or None
          - "meta" : dict with extra info (evidence, timing, etc.)
    """
    samples, param_names = _nautilus_points_to_array(points)
    log_w = np.asarray(log_w, float).ravel()
    log_l = np.asarray(log_l, float).ravel()

    n_total = samples.shape[0]
    if log_w.shape[0] != n_total or log_l.shape[0] != n_total:
        raise ValueError("Inconsistent shapes between points, log_w, and log_l.")

    rng = _get_rng(random_seed)

    # Resample if requested
    if n_samples is not None:
        n_samples = int(n_samples)
        if use_weights:
            # Importance resampling using the NAUTILUS weights
            w = np.exp(log_w - np.max(log_w))
            w_sum = w.sum()
            if w_sum <= 0.0 or not np.isfinite(w_sum):
                # Fall back to uniform if something is off
                p = None
            else:
                p = w / w_sum
            idx = rng.choice(n_total, size=n_samples, replace=True, p=p)
        else:
            # Uniform subsampling from the available points
            replace = n_samples > n_total
            idx = rng.choice(n_total, size=n_samples, replace=replace)

        samples = samples[idx]
        log_l = log_l[idx]
        # After resampling, treat samples as unweighted
        log_w_out = np.zeros(samples.shape[0], dtype=float)
    else:
        # Keep the original weighted set
        log_w_out = log_w

    # Try to get log-evidence if available
    log_evidence = None
    if hasattr(sampler, "logz"):
        try:
            log_evidence = float(np.array(sampler.logz).ravel()[-1])
        except Exception:
            log_evidence = None

    extra = dict(
        n_raw_samples=int(n_total),
        n_samples=int(samples.shape[0]),
        log_evidence=log_evidence,
        timing=getattr(sampler, "timing", None),
        raw_log_weights_shape=log_w.shape,
    )

    # Keep the original log-weights also in the metadata, for advanced use
    extra["raw_log_weights"] = log_w

    return _standard_posterior_dict(
        engine="nautilus",
        samples=samples,
        log_likelihood=log_l,
        log_weights=log_w_out,
        param_names=param_names,
        extra=extra,
    )
def get_emcee_posterior(
    chain,
    log_prob_chain,
    param_names,
    burnin_fraction=0.5,
    thin=1,
    n_samples=None,
    random_seed=None,
):
    """
    Convert an emcee chain into a standard posterior dictionary.

    Parameters
    ----------
    chain : array-like
        MCMC chain with shape (n_steps, n_walkers, ndim), as returned by
        `sampler.get_chain()` from `run_emcee`.
    log_prob_chain : array-like
        Log-posterior values with shape (n_steps, n_walkers), as returned by
        `sampler.get_log_prob()` from `run_emcee`.
    param_names : list[str]
        Names of the parameters, in the same order as the chain's last axis.
    burnin_fraction : float
        Fraction of initial steps to discard as burn-in (0.0–1.0).
    thin : int
        Thinning factor: keep every `thin`-th step after burn-in.
    n_samples : int or None
        If None: keep all flattened samples after burn-in and thinning.
        If int: draw `n_samples` samples (without replacement if possible,
        otherwise with replacement).
    random_seed : int or None
        Seed for optional subsampling.

    Returns
    -------
    posterior : dict
        Standardised posterior dictionary with keys:
          - "engine" = "emcee"
          - "samples" : (n_samples, ndim)
          - "log_likelihood" : (n_samples,)
          - "log_weights" : zeros((n_samples,))
          - "param_names" : list[str]
          - "meta" : dict with extra info (n_raw_samples, etc.)
    """
    chain = np.asarray(chain, float)
    log_prob_chain = np.asarray(log_prob_chain, float)

    if chain.ndim != 3:
        raise ValueError("chain must have shape (n_steps, n_walkers, ndim).")
    if log_prob_chain.ndim != 2:
        raise ValueError("log_prob_chain must have shape (n_steps, n_walkers).")

    n_steps, n_walkers, ndim = chain.shape
    if log_prob_chain.shape != (n_steps, n_walkers):
        raise ValueError("log_prob_chain shape must match (n_steps, n_walkers).")

    if param_names is not None and len(param_names) != ndim:
        raise ValueError("len(param_names) must match the chain's last dimension.")

    # Burn-in + thinning
    burnin = int(np.clip(int(burnin_fraction * n_steps), 0, n_steps - 1))
    if thin is None or thin < 1:
        thin = 1

    chain_post = chain[burnin::thin, :, :]          # (n_post_steps, n_walkers, ndim)
    logp_post = log_prob_chain[burnin::thin, :]     # (n_post_steps, n_walkers)

    # Flatten (step, walker) -> sample axis
    samples = chain_post.reshape(-1, ndim)
    log_l = logp_post.reshape(-1)
    n_total = samples.shape[0]

    rng = _get_rng(random_seed)

    # Optional subsampling
    if n_samples is not None:
        n_samples = int(n_samples)
        replace = n_samples > n_total
        idx = rng.choice(n_total, size=n_samples, replace=replace)
        samples = samples[idx]
        log_l = log_l[idx]

    # emcee yields unweighted samples -> log_weights = 0
    log_w = np.zeros(samples.shape[0], dtype=float)

    extra = dict(
        n_raw_steps=int(n_steps),
        n_walkers=int(n_walkers),
        ndim=int(ndim),
        burnin_fraction=float(burnin_fraction),
        thin=int(thin),
        n_raw_samples=int(n_steps * n_walkers),
        n_samples=int(samples.shape[0]),
    )

    return _standard_posterior_dict(
        engine="emcee",
        samples=samples,
        log_likelihood=log_l,
        log_weights=log_w,
        param_names=param_names,
        extra=extra,
    )



__all__ = [
    "make_safe_loss",
    "adam_preopt",
    "adam_preopt_vmap",
    "run_multistart",
    "run_multistart_vmap",
    "tnorm",
    "build_nautilus_prior",
    "make_paramdict_to_kwargs",
    "build_gaussian_loglike",
    "build_nautilus_prior_and_loglike",
    "build_emcee_prior",
    "build_emcee_logprob",
    "draw_emcee_initial_positions",
    "run_emcee",
    "run_nautilus",
    "load_posterior_from_checkpoint",
    "load_multistart_summary",
    "build_svi",
    "run_svi_chains",
]
