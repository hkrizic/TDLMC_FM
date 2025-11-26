# tdlmc_model.py
"""
Model & data-loading utilities for TDLMC strong-lensing with herculens.

This collects everything that is purely about the *model*:
- loading the TDLMC image / PSF / noise
- building pixel / PS grids and instrument objects
- detecting point-source images
- defining the NumPyro ProbModel with the EPL+SHEAR mass model,
  SERSIC_ELLIPSE lens & source light, and IMAGE_POSITIONS point sources.

Typical usage in a notebook:

    from tdlmc_model import setup_tdlmc_lens

    cfg = setup_tdlmc_lens(base=".",
                           rung=2, code_id=1, seed=122,
                           n_ps_detect=4)

    prob_model  = cfg["prob_model"]
    lens_image  = cfg["lens_image"]
    plotter     = cfg["plotter"]
    img         = cfg["img"]
    noise_map   = cfg["noise_map"]
    outdir      = cfg["outdir"]

You can then pass `prob_model`, `img`, `noise_map`, `outdir`
to the inference utilities in `tdlmc_inference.py`.
"""

import os
from typing import Dict, Tuple

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from astropy.io import fits
from scipy.ndimage import maximum_filter

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.LightModel.light_model import LightModel
from herculens.MassModel.mass_model import MassModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.LensImage.lens_image import LensImage
from herculens.Analysis.plot import Plotter
from herculens.Inference.ProbModel.numpyro import NumpyroModel


def tdlmc_paths(base: str, rung: int, code_id: int, seed: int) -> Tuple[str, str]:
    """
    Return (folder, outdir) paths for the drizzled TDLMC image and results.

    folder:  .../TDC/rung{rung}/code{code_id}/f160w-seed{seed}/drizzled_image
    outdir:  .../TDC_results/rung{rung}/code{code_id}/f160w-seed{seed}
    """
    code = f"code{code_id}"
    folder = os.path.join(
        base,
        f"TDC/rung{rung}/{code}/f160w-seed{seed}/drizzled_image",
    )
    outdir = os.path.join(
        base,
        f"TDC_results/rung{rung}/{code}/f160w-seed{seed}",
    )
    os.makedirs(outdir, exist_ok=True)
    return folder, outdir


def load_tdlmc_image(folder: str):
    """Load image, PSF kernel and noise map from a TDLMC drizzled_image folder."""
    img = fits.getdata(os.path.join(folder, "lens-image.fits"), header=False).astype(
        np.float64
    )
    psf_kernel = fits.getdata(
        os.path.join(folder, "psf.fits"), header=False
    ).astype(np.float64)
    noise_map = fits.getdata(
        os.path.join(folder, "noise_map.fits"), header=False
    ).astype(np.float64)
    return img, psf_kernel, noise_map


def make_pixel_grids(
    img: np.ndarray,
    pix_scl: float = 0.08,
    ps_oversample: int = 2,
):
    """
    Build the main PixelGrid and a supersampled grid for point sources.

    Returns (pixel_grid, ps_grid, xgrid, ygrid, pix_scl).
    """
    npix_y, npix_x = img.shape
    assert npix_x == npix_y, "Expect a square image."
    npix = npix_x

    half_size = npix * pix_scl / 2.0
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2.0
    transform_pix2angle = pix_scl * np.eye(2)

    pixel_grid = PixelGrid(
        nx=npix,
        ny=npix,
        ra_at_xy_0=ra_at_xy_0,
        dec_at_xy_0=dec_at_xy_0,
        transform_pix2angle=transform_pix2angle,
    )

    # High-res PS grid for subpixel point-source positions
    ps_grid_npix = ps_oversample * npix + 1  # default: 2*npix + 1
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.0
    ps_grid_ra0 = ps_grid_dec0 = -ps_grid_half_size + ps_grid_pix_scl / 2.0
    ps_grid = PixelGrid(
        nx=ps_grid_npix,
        ny=ps_grid_npix,
        ra_at_xy_0=ps_grid_ra0,
        dec_at_xy_0=ps_grid_dec0,
        transform_pix2angle=ps_grid_pix_scl * np.eye(2),
    )

    xgrid, ygrid = pixel_grid.pixel_coordinates

    return pixel_grid, ps_grid, xgrid, ygrid, pix_scl


def make_lens_image(
    img: np.ndarray,
    psf_kernel: np.ndarray,
    noise_map: np.ndarray,
    pixel_grid: PixelGrid,
    ps_grid: PixelGrid,
    supersampling_factor: int = 5,
    convolution_type: str = "jax_scipy_fft",
):
    """
    Build PSF, Noise, Mass/Light models and LensImage.

    Returns
    -------
    lens_image, noise, psf, mass_model, lens_light_model, source_light_model, point_source_model
    """
    npix_y, npix_x = img.shape

    psf = PSF(psf_type="PIXEL", kernel_point_source=psf_kernel)
    noise = Noise(npix_x, npix_y, noise_map=noise_map)

    mass_model = MassModel(["EPL", "SHEAR"])
    lens_light_model = LightModel(["SERSIC_ELLIPSE"])
    source_light_model = LightModel(["SERSIC_ELLIPSE"])
    point_source_model = PointSourceModel(["IMAGE_POSITIONS"], mass_model, ps_grid)

    kwargs_numerics = dict(
        supersampling_factor=supersampling_factor,
        convolution_type=convolution_type,
        supersampling_convolution=False,
    )

    lens_image = LensImage(
        pixel_grid,
        psf,
        noise_class=noise,
        lens_mass_model_class=mass_model,
        lens_light_model_class=lens_light_model,
        source_model_class=source_light_model,
        point_source_model_class=point_source_model,
        kwargs_numerics=kwargs_numerics,
    )

    return (
        lens_image,
        noise,
        psf,
        mass_model,
        lens_light_model,
        source_light_model,
        point_source_model,
    )


def make_plotter(img: np.ndarray) -> Plotter:
    """Convenience helper to build a Plotter with sensible vmin/vmax for a given image."""
    vmin = max(1e-6, float(np.percentile(img, 0.5)))
    vmax = float(np.percentile(img, 99.7))
    plotter = Plotter(flux_vmin=vmin, flux_vmax=vmax, res_vmax=5)
    plotter.set_data(img)
    return plotter


def detect_ps_images_centered(
    img: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    n_wanted: int = 4,
    lens_mask_radius: float = 0.5,  # arcsec: fixed mask for central lens
    local_win: int = 3,  # window size for local maxima (odd)
    min_peak_frac: float = 0.15,  # keep peaks >=15% of global max
    min_sep: float = 0.18,  # arcsec: enforce min separation
):
    """
    Detect bright point-source images, assuming lens galaxy is centered.
    1) Masks fixed circular region at (0,0)
    2) Finds local maxima outside that region
    3) Keeps brightest peaks with min angular separation

    Returns
    -------
    peaks_px : list[(y, x)]
    x0s      : np.ndarray of R.A. positions (arcsec)
    y0s      : np.ndarray of Dec positions (arcsec)
    peak_vals: np.ndarray of peak flux values
    """
    ny, nx = img.shape

    # mask circular lens region around (0,0) in angle-space
    dist_ang = np.hypot(xgrid, ygrid)
    mask_lens = dist_ang < lens_mask_radius

    # local maxima
    local_max = maximum_filter(img, size=local_win, mode="nearest")
    is_local_max = img == local_max

    # brightness threshold
    peak_max = float(img[~mask_lens].max())
    thr = min_peak_frac * peak_max

    cand = np.vstack(np.where(is_local_max & (~mask_lens) & (img >= thr))).T
    if cand.size == 0:
        return [], np.array([]), np.array([]), np.array([])

    cand_vals = img[cand[:, 0], cand[:, 1]]
    order = np.argsort(cand_vals)[::-1]
    cand = cand[order]

    picked, picked_ra, picked_dec, picked_flux = [], [], [], []
    for (yy, xx) in cand:
        ra = float(xgrid[yy, xx])
        dec = float(ygrid[yy, xx])
        fl = float(img[yy, xx])
        ok = True
        for pra, pdec in zip(picked_ra, picked_dec):
            if np.hypot(ra - pra, dec - pdec) < min_sep:
                ok = False
                break
        if not ok:
            continue

        picked.append((int(yy), int(xx)))
        picked_ra.append(ra)
        picked_dec.append(dec)
        picked_flux.append(fl)
        if len(picked) >= n_wanted:
            break

    return picked, np.array(picked_ra), np.array(picked_dec), np.array(picked_flux)


class ProbModel(NumpyroModel):
    """
    NumPyro probabilistic model for the TDLMC lens:

    - Mass: EPL + external shear
    - Lens light: SERSIC_ELLIPSE
    - Source light: SERSIC_ELLIPSE
    - Point sources: IMAGE_POSITIONS (ra/dec/amp for each image)
    """

    def __init__(
        self,
        lens_image: LensImage,
        img: np.ndarray,
        noise_map: np.ndarray,
        xgrid: np.ndarray,
        ygrid: np.ndarray,
        x0s: np.ndarray,
        y0s: np.ndarray,
        peak_vals: np.ndarray,
    ):
        super().__init__()
        self.lens_image = lens_image
        self.data = jnp.asarray(img)
        self.noise_map = jnp.asarray(noise_map)
        self.xgrid = jnp.asarray(xgrid)
        self.ygrid = jnp.asarray(ygrid)
        self.x0s = jnp.asarray(x0s)
        self.y0s = jnp.asarray(y0s)
        self.peak_vals = jnp.asarray(peak_vals)

    def model(self):
        # Work in numpy for simple statistics
        img = np.asarray(self.data)
        xgrid = np.asarray(self.xgrid)
        ygrid = np.asarray(self.ygrid)
        x0s = self.x0s
        y0s = self.y0s
        peak_vals = self.peak_vals

        # simple centroid for lens center prior
        p5, p995 = np.percentile(img, [5.0, 99.5])
        clip = np.clip(img, p5, p995)
        w = np.maximum(clip - clip.min(), 0.0)
        W = w.sum() + 1e-12
        cx = float((w * xgrid).sum() / W)
        cy = float((w * ygrid).sum() / W)

        # --- Mass ---
        lens_center_x = numpyro.sample("lens_center_x", dist.Normal(cx, 0.3))
        lens_center_y = numpyro.sample("lens_center_y", dist.Normal(cy, 0.3))
        theta_E = numpyro.sample("lens_theta_E", dist.Uniform(0.3, 2.2))
        e1 = numpyro.sample("lens_e1", dist.Uniform(-0.4, 0.4))
        e2 = numpyro.sample("lens_e2", dist.Uniform(-0.4, 0.4))
        gamma = numpyro.sample("lens_gamma", dist.Uniform(1.2, 2.8))
        gamma1 = numpyro.sample("lens_gamma1", dist.Uniform(-0.3, 0.3))
        gamma2 = numpyro.sample("lens_gamma2", dist.Uniform(-0.3, 0.3))

        # --- Lens light ---
        amp90 = float(np.percentile(img, 90.0))
        light_amp_L = numpyro.sample(
            "light_amp_L", dist.LogNormal(np.log(max(amp90, 1e-6)), 1.0)
        )
        light_Re_L = numpyro.sample("light_Re_L", dist.Uniform(0.05, 2.5))
        light_n_L = numpyro.sample("light_n_L", dist.Uniform(0.7, 5.5))
        light_e1_L = numpyro.sample("light_e1_L", dist.Uniform(-0.6, 0.6))
        light_e2_L = numpyro.sample("light_e2_L", dist.Uniform(-0.6, 0.6))

        # --- Source light ---
        amp70 = float(np.percentile(img, 70.0))
        light_amp_S = numpyro.sample(
            "light_amp_S", dist.LogNormal(np.log(max(amp70, 3e-6)), 1.2)
        )
        light_Re_S = numpyro.sample("light_Re_S", dist.Uniform(0.03, 1.2))
        light_n_S = numpyro.sample("light_n_S", dist.Uniform(0.5, 4.5))
        light_e1_S = numpyro.sample("light_e1_S", dist.Uniform(-0.8, 0.8))
        light_e2_S = numpyro.sample("light_e2_S", dist.Uniform(-0.8, 0.8))
        src_center_x = numpyro.sample("src_center_x", dist.Normal(0.0, 0.6))
        src_center_y = numpyro.sample("src_center_y", dist.Normal(0.0, 0.6))

        # --- Point sources: priors centered on detected peaks ---
        x_image = numpyro.sample(
            "x_image",
            dist.Independent(dist.Normal(x0s, 0.2), 1),
        )
        y_image = numpyro.sample(
            "y_image",
            dist.Independent(dist.Normal(y0s, 0.2), 1),
        )
        ps_amp = numpyro.sample(
            "ps_amp",
            dist.Independent(
                dist.LogNormal(jnp.log(jnp.maximum(peak_vals, 1e-6)), 0.6),
                1,
            ),
        )

        kwargs_lens = [
            dict(
                theta_E=theta_E,
                e1=e1,
                e2=e2,
                center_x=lens_center_x,
                center_y=lens_center_y,
                gamma=gamma,
            ),
            dict(gamma1=gamma1, gamma2=gamma2, ra_0=0.0, dec_0=0.0),
        ]
        kwargs_lens_light = [
            dict(
                amp=light_amp_L,
                R_sersic=light_Re_L,
                n_sersic=light_n_L,
                e1=light_e1_L,
                e2=light_e2_L,
                center_x=lens_center_x,
                center_y=lens_center_y,
            )
        ]
        kwargs_source = [
            dict(
                amp=light_amp_S,
                R_sersic=light_Re_S,
                n_sersic=light_n_S,
                e1=light_e1_S,
                e2=light_e2_S,
                center_x=src_center_x,
                center_y=src_center_y,
            )
        ]
        kwargs_point = [dict(ra=x_image, dec=y_image, amp=ps_amp)]

        model_img = self.lens_image.model(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

        numpyro.sample(
            "obs",
            dist.Independent(dist.Normal(model_img, self.noise_map), 2),
            obs=self.data,
        )

    def params2kwargs(self, params: Dict):
        """
        Convert a constrained parameter dict (as returned by prob_model.constrain)
        into kwargs for LensImage.model, so optimisation and sampling code can
        re-use the forward model without re-implementing the mapping.
        """
        kwargs_lens = [
            dict(
                theta_E=params["lens_theta_E"],
                e1=params["lens_e1"],
                e2=params["lens_e2"],
                center_x=params["lens_center_x"],
                center_y=params["lens_center_y"],
                gamma=params["lens_gamma"],
            ),
            dict(
                gamma1=params["lens_gamma1"],
                gamma2=params["lens_gamma2"],
                ra_0=0.0,
                dec_0=0.0,
            ),
        ]
        kwargs_lens_light = [
            dict(
                amp=params["light_amp_L"],
                R_sersic=params["light_Re_L"],
                n_sersic=params["light_n_L"],
                e1=params["light_e1_L"],
                e2=params["light_e2_L"],
                center_x=params["lens_center_x"],
                center_y=params["lens_center_y"],
            )
        ]
        kwargs_source = [
            dict(
                amp=params["light_amp_S"],
                R_sersic=params["light_Re_S"],
                n_sersic=params["light_n_S"],
                e1=params["light_e1_S"],
                e2=params["light_e2_S"],
                center_x=params["src_center_x"],
                center_y=params["src_center_y"],
            )
        ]
        kwargs_point = [
            dict(
                ra=params["x_image"],
                dec=params["y_image"],
                amp=params["ps_amp"],
            )
        ]
        return dict(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
            kwargs_point_source=kwargs_point,
        )

    # --- New helper methods for multistart χ² tracking ---

    def model_image_from_params(self, params: Dict):
        """
        Forward model for a given constrained parameter dict.

        Parameters
        ----------
        params : dict
            Constrained parameters as returned by prob_model.constrain.

        Returns
        -------
        model_img : jnp.ndarray
            Model image on the data grid.
        """
        kwargs = self.params2kwargs(params)
        model_img = self.lens_image.model(**kwargs)
        return jnp.asarray(model_img)

    def reduced_chi2(self, params: Dict, n_params: int | None = None) -> float:
        """
        Compute reduced chi^2 for a given set of constrained parameters.

        Parameters
        ----------
        params : dict
            Constrained parameters as returned by prob_model.constrain.
        n_params : int or None
            Number of free parameters in the fit. If None, len(params) is used.
            You can set this to match the definition used in plot_bestfit_model.

        Returns
        -------
        float
            Reduced chi^2 value.
        """
        model_img = self.model_image_from_params(params)
        resid = (self.data - model_img) / self.noise_map
        chi2 = jnp.sum(resid**2)

        n_pix = self.data.size
        if n_params is None:
            n_params = len(params)
        dof = max(int(n_pix - n_params), 1)
        chi2_red = chi2 / dof
        return float(chi2_red)


def setup_tdlmc_lens(
    base: str,
    rung: int,
    code_id: int,
    seed: int,
    n_ps_detect: int = 4,
    pix_scl: float = 0.08,
    ps_oversample: int = 2,
    lens_mask_radius: float = 0.5,
    local_win: int = 3,
    min_peak_frac: float = 0.15,
    min_sep: float = 0.18,
    supersampling_factor: int = 5,
    convolution_type: str = "jax_scipy_fft",
) -> Dict:
    """
    High-level convenience wrapper that builds everything needed for inference.

    Returns a dict with keys:
      - folder, outdir
      - img, psf_kernel, noise_map
      - pixel_grid, ps_grid, xgrid, ygrid
      - peaks_px, x0s, y0s, peak_vals
      - lens_image, plotter, prob_model
    """
    folder, outdir = tdlmc_paths(base, rung, code_id, seed)
    img, psf_kernel, noise_map = load_tdlmc_image(folder)
    pixel_grid, ps_grid, xgrid, ygrid, pix_scl = make_pixel_grids(
        img, pix_scl=pix_scl, ps_oversample=ps_oversample
    )
    peaks_px, x0s, y0s, peak_vals = detect_ps_images_centered(
        img,
        xgrid,
        ygrid,
        n_wanted=n_ps_detect,
        lens_mask_radius=lens_mask_radius,
        local_win=local_win,
        min_peak_frac=min_peak_frac,
        min_sep=min_sep,
    )
    (
        lens_image,
        noise,
        psf,
        mass_model,
        lens_light_model,
        source_light_model,
        point_source_model,
    ) = make_lens_image(
        img,
        psf_kernel,
        noise_map,
        pixel_grid,
        ps_grid,
        supersampling_factor=supersampling_factor,
        convolution_type=convolution_type,
    )
    plotter = make_plotter(img)
    prob_model = ProbModel(
        lens_image=lens_image,
        img=img,
        noise_map=noise_map,
        xgrid=xgrid,
        ygrid=ygrid,
        x0s=x0s,
        y0s=y0s,
        peak_vals=peak_vals,
    )
    return dict(
        folder=folder,
        outdir=outdir,
        img=img,
        psf_kernel=psf_kernel,
        noise_map=noise_map,
        pixel_grid=pixel_grid,
        ps_grid=ps_grid,
        xgrid=xgrid,
        ygrid=ygrid,
        peaks_px=peaks_px,
        x0s=x0s,
        y0s=y0s,
        peak_vals=peak_vals,
        lens_image=lens_image,
        plotter=plotter,
        prob_model=prob_model,
        pix_scl=pix_scl,
    )


__all__ = [
    "tdlmc_paths",
    "load_tdlmc_image",
    "make_pixel_grids",
    "make_lens_image",
    "make_plotter",
    "detect_ps_images_centered",
    "ProbModel",
    "setup_tdlmc_lens",
]