__author__ = 'martin-millon'

import re
import ast
import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves, tree_map

import numpy as np
import astropy.units as u

import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

from astropy import constants as cst
import scipy

import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm, LogNorm, TwoSlopeNorm
from herculens.Util import plot_util

def cast_ints_to_floats_in_dict(d):
    """
    Recursively traverse dictionary and:
      - cast int to float
      - leave str and float unchanged
      - recursively check dicts and lists
    """
    if isinstance(d, dict):
        return {k: cast_ints_to_floats_in_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [cast_ints_to_floats_in_dict(item) for item in d]
    elif isinstance(d, int):
        return float(d)
    elif isinstance(d, (str, float)):
        return d
    elif d is None:
        return None
    else:
        raise TypeError(f"Unsupported type: {type(d)} with value: {d}")

def parse_lens_info_file(filepath):
    def extract_value(pattern, text, eval_type=float, default=None):
        match = re.search(pattern, text)
        if match:
            try:
                return eval_type(match.group(1))
            except:
                return default
        return default

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse the dictionary
    lens_data = {
        "units": {
            "angle": "arcseconds",
            "phi_G": "radians (from x-axis, anticlockwise)"
        },
        "cosmology": {
            "model": "FlatLambdaCDM",
            "Om": extract_value(r"Om\s*=\s*([0-9.]+)", content),
            "H0": extract_value(r"H0:\s*([0-9.]+)", content)
        },
        "pixel_size": {
            "before_drizzle": extract_value(r"Pixel size is ([0-9.]+)''", content),
            "after_drizzle": extract_value(r"and ([0-9.]+)'' after drizzle", content)
        },
        "time_delay_distance": extract_value(r"Time delay distance:.*?([0-9.]+)Mpc", content),
        "time_delays_BCD_minus_A": ast.literal_eval(extract_value(r"Time delay of BCD - A\s*:\s*array\((.*?)\)", content, str, "[]")),
        "zeropoint_AB": extract_value(r"Zeropoint of filter.*?:\s*([0-9.]+)", content),
        "redshifts": {
            "lens": None,
            "source": None
        },
        "lens_mass_model": {
            "SPEMD": None,
            "shear": None
        },
        "lens_light": None,
        "source_light": {
            "host_name": None,
            "center_pos": None,
            "mag": None,
            "R_eff": None
        },
        "AGN_light": {
            "source_plane": {},
            "image_plane": {}
        },
        "magnitudes": {},
        "velocity_dispersion": extract_value(r"Measured velocity dispersion.*?:\s*([0-9.]+)", content),
        "kappa_ext": extract_value(r"kappa_ext.*?:\s*([0-9.]+)", content),
        "time_delays_with_kappa_ext": ast.literal_eval(extract_value(r"Time delay with external kappa.*?:\s*array\((.*?)\)", content, str, "[]"))
    }

    # Parse redshifts
    redshift_match = re.search(r"Lens/Source redshift:\s*\[([0-9.,\s]+)\]", content)
    if redshift_match:
        z_l, z_s = map(float, redshift_match.group(1).split(","))
        lens_data["redshifts"]["lens"] = z_l
        lens_data["redshifts"]["source"] = z_s

    # Parse SPEMD lens mass model
    match = re.search(r"SPEMD:(\{.*?\})", content, re.DOTALL)
    if match:
        lens_data["lens_mass_model"]["SPEMD"] = ast.literal_eval(match.group(1))

    # Parse shear
    match = re.search(r"Shear:\s*\((\{.*?\}),\s*(\{.*?\})\)", content)
    if match:
        shear = ast.literal_eval(match.group(1))
        lens_data["lens_mass_model"]["shear"] = shear

    # Parse lens light
    match = re.search(r"Lens light:\s*\n\s*(\{.*?\})", content, re.DOTALL)
    if match:
        lens_data["lens_light"] = ast.literal_eval(match.group(1))

    # Parse host galaxy info
    match = re.search(r"Host galaxy name:\s*(\S+)\s*CenterPos:\s*array\((.*?)\)", content)
    if match:
        lens_data["source_light"]["host_name"] = match.group(1)
        lens_data["source_light"]["center_pos"] = ast.literal_eval(match.group(2))

    lens_data["source_light"]["mag"] = extract_value(r"Host mag:\s*([0-9.]+)", content)
    lens_data["source_light"]["R_eff"] = extract_value(r"Host R_eff:\s*([0-9.]+)", content)

    # Parse AGN light
    lens_data["AGN_light"]["source_plane"]["position"] = list(map(float, re.findall(r"AGN position in source plane:\s*([0-9.\-]+),\s*([0-9.\-]+)", content)[0]))
    lens_data["AGN_light"]["source_plane"]["amplitude"] = extract_value(r"AGN amplitude in source plane:\s*([0-9.]+)", content)

    match = re.search(r"AGN position in image plane:\s*x:\s*array\((.*?)\)\s*y:\s*array\((.*?)\)", content, re.DOTALL)
    if match:
        lens_data["AGN_light"]["image_plane"]["x"] = ast.literal_eval(match.group(1))
        lens_data["AGN_light"]["image_plane"]["y"] = ast.literal_eval(match.group(2))

    match = re.search(r"AGN amplitude in image plane:\s*array\((.*?)\)", content)
    if match:
        lens_data["AGN_light"]["image_plane"]["amplitude"] = ast.literal_eval(match.group(1))

    lens_data["magnitudes"]["host_galaxy_image_plane"] = extract_value(r"Host galaxy mag in the image plane:\s*([0-9.]+)", content)
    lens_data["magnitudes"]["AGN_total_image_plane"] = extract_value(r"AGN total mag in the image plane:\s*([0-9.]+)", content)

    return cast_ints_to_floats_in_dict(lens_data)



def parse_good_team_lens_info_file(filepath):
    with open(filepath, "r") as f:
        content = f.read()

    def extract_value(pattern, eval_type=float, default=None):
        match = re.search(pattern, content)
        if match:
            try:
                return eval_type(match.group(1))
            except:
                return default
        return default

    lens_data = {
        "pixel_size": {
            "before_drizzle": extract_value(r"Pixel size is ([0-9.]+)''"),
            "after_drizzle": extract_value(r"and ([0-9.]+)'' after drizzle")
        },
        "zeropoint_AB": extract_value(r"Zeropoint of filter \(AB system\):\s*([0-9.]+)"),
        "redshifts": {
            "lens": None,
            "source": None
        },
        "external_convergence": {
            "kappa_ext": None,
            "kappa_ext_error": None
        },
        "velocity_dispersion": {
            "value": None,
            "error": None
        },
        "time_delays_BCD_minus_A": {
            "values": None,
            "errors": None
        }
    }

    # Redshifts
    match = re.search(r"Lens/Source redshift:\s*\[([0-9.,\s]+)\]", content)
    if match:
        z_lens, z_source = map(float, match.group(1).split(","))
        lens_data["redshifts"]["lens"] = z_lens
        lens_data["redshifts"]["source"] = z_source

    # External convergence
    match = re.search(r"External Convergence: Kext=\s*([-\d.]+)\s*\+/-\s*([0-9.]+)", content)
    if match:
        lens_data["external_convergence"]["kappa_ext"] = float(match.group(1))
        lens_data["external_convergence"]["kappa_ext_error"] = float(match.group(2))

    # Velocity dispersion
    match = re.search(r"Measured velocity dispersion:\s*([0-9.]+)km/s, error level:\s*([0-9.]+)km/s", content)
    if match:
        lens_data["velocity_dispersion"]["value"] = float(match.group(1))
        lens_data["velocity_dispersion"]["error"] = float(match.group(2))

    # Time delays
    match = re.search(r"Time delay of BCD - A\s*:\s*array\((.*?)\)days,\s*error level:\s*array\((.*?)\)days", content)
    if match:
        delays = ast.literal_eval(f"[{match.group(1)}]")
        delay_errors = ast.literal_eval(f"[{match.group(2)}]")
        lens_data["time_delays_BCD_minus_A"]["values"] = delays[0]
        lens_data["time_delays_BCD_minus_A"]["errors"] = delay_errors[0]

    return lens_data

def compute_D_dt(z_lens, z_source, cosmology):
    """
    Compute the time-delay distance D_dt in Mpc given lens and source redshifts.
    Parameters
    ----------
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source  
    cosmology : astropy.cosmology.Cosmology
        Cosmology object (e.g. FlatLambdaCDM)   
    Returns
    -------
    D_dt : float
        Time-delay distance in Mpc
    """

    D_l = cosmology.angular_diameter_distance(z_lens).to(u.Mpc).value  # in Mpc
    D_s = cosmology.angular_diameter_distance(z_source).to(u.Mpc).value 
    D_ls = cosmology.angular_diameter_distance_z1z2(z_lens, z_source).to(u.Mpc).value 

    # Time-delay distance
    D_dt = (1 + z_lens) * D_l * D_s / D_ls  # in Mpc

    return D_dt  # in Mpc

def predict_time_delay(delta_phi, z_lens, z_source, cosmology=None, D_dt=None):
    """
    Predict time delay (in days) given Fermat potential difference and cosmology.

    Parameters
    ----------
    delta_phi : float
        Fermat potential difference between two images in arcsec**2
    z_lens : float
        Redshift of the lens
    z_source : float
        Redshift of the source
    cosmology : astropy.cosmology.Cosmology
        Cosmology object (e.g. FlatLambdaCDM)
    D_dt : float, optional
        Time-delay distance in Mpc. If not provided, it will be calculated from cosmology

    Returns
    -------
    delta_t_days : float
        Predicted time delay in days
    """
    Mpc_to_m = 3.085677581491367e22  # 1 Mpc in meters
    s_to_days = 86400  # seconds in a day

    if cosmology is None and D_dt is None:
        raise ValueError("Either cosmology or D_dt must be provided.")
    if D_dt is None:
        D_dt = compute_D_dt(z_lens, z_source, cosmology)

    delta_phi = (delta_phi * u.arcsec**2).to(u.rad**2).value # convert to astropy units
    # Time delay in seconds
    delta_t = (D_dt / cst.c.to(u.m / u.s).value * delta_phi) * Mpc_to_m / s_to_days  # convert to days

    return delta_t


def model_Ddt_with_pred_samples(pred_fermat_pot_samples, obs_delays, obs_errors, z_lens, z_source):
    """
    Model D_dt marginalizing over predicted delay samples (Monte Carlo estimate).

    Parameters
    ----------
    pred_delay_samples : array (S, N)
        N_predicted_samples × N_delays
    obs_delays : array (N,)
        Observed time delays
    obs_errors : array (N,)
        Observational uncertainties
    """
    Ddt = numpyro.sample("D_dt", dist.Uniform(0.0, 15000.0))

    # Shape: (S, N)
    predicted_delays = predict_time_delay(pred_fermat_pot_samples, z_lens, z_source, D_dt=Ddt)

    # Monte Carlo marginalization (logsumexp over S)
    def log_prob_single(pred):
        return jnp.sum(dist.Normal(pred, obs_errors).log_prob(obs_delays))

    log_liks = jax.vmap(log_prob_single)(predicted_delays)
    log_marginal = jax.scipy.special.logsumexp(log_liks) - jnp.log(predicted_delays.shape[0])

    numpyro.factor("log_marginal_likelihood", log_marginal)

def integrand(z, omegaM, omegaL):
	omegaM = float(omegaM)
	omegaL = float(omegaL)
	denom = (1.0-omegaM-omegaL)*(1.0+z)**2 + omegaM*(1.0+z)**3 + omegaL
	if denom < 0:
		raise RuntimeError("'denom < 0' in integrand")
	return 1.0 / np.sqrt(denom)

def Ddt_2_H0(Ddt, z_lens, z_source, Omega_M, Omega_L):
    c = cst.c
    A = (scipy.integrate.quad(integrand, 0.0, z_source, args=(Omega_M, Omega_L))[0] *
         scipy.integrate.quad(integrand, 0, z_lens, args=(Omega_M, Omega_L))[0]) / \
         scipy.integrate.quad(integrand, z_lens, z_source, args=(Omega_M, Omega_L))[0]

    return (c.to('km/s').value*A)/Ddt

def Dd_2_H0(Dd,z_lens, Omega_M, Omega_L) :
    c = cst.c
    A = scipy.integrate.quad(integrand, 0.0, z_lens, args=(Omega_M, Omega_L))[0]

    return (c.to('km/s').value*A)/(Dd*(1+z_lens))


def visualize_initial_guess(init_params, lens_image, prob_model, data, plotter):
    # Evaluate the model
    initial_model = lens_image.model(**prob_model.params2kwargs(init_params))

    # visualize initial guess
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    ax = axes[0]
    ax.set_title("Initial guess model")
    im = ax.imshow(initial_model, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)
    ax = axes[1]
    ax.set_title("Data")
    im = ax.imshow(data, origin='lower', norm=plotter.norm_flux, cmap=plotter.cmap_flux)
    plot_util.nice_colorbar(im)
    ax = axes[2]
    ax.set_title("Difference")
    im = ax.imshow(initial_model - data, origin='lower', norm=TwoSlopeNorm(0), cmap=plotter.cmap_res)
    plot_util.nice_colorbar(im)
    fig.tight_layout()

    return fig, axes

@jax.jit
def get_value_from_index(xs, i):
    """useful helper function to get the i-th element from a pytree"""
    return jax.tree.map(lambda x: x[i], xs)

def is_batched_pytree(params):
    """ Check if the pytree contains batched parameters.
    A batched pytree has leaves that are arrays with shape[0] > 1
    """
    leaves = tree_leaves(params)
    return all(hasattr(x, 'shape') and x.ndim > 0 and x.shape[0] > 1 for x in leaves)