"""
Algorithm based on Kao et al. (2005),
"A Software tool for measurement of the modulation transfer function"

Drawing ROIs around the edges of the MTF tool is automated by the get_labelled_rois
function.
"""

from pathlib import Path
from enum import Enum
from dataclasses import dataclass
import warnings
import pydicom
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from numba import njit, prange
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor, LinearRegression, RANSACRegressor
from .errors import ESFAngleError
from .roifind import get_labelled_rois, detect_edge_with_outlier_replacement
from .dcmutils import preprocess_dcm


def get_cutoff_frequency(
    spatial_frequencies: np.array,
    mtf_values: np.array,
    cutoff_point: float,
    mtf_threshold: float = 0.01,
) -> float:
    """
    Finds the cutoff frequency at a specified MTF value using interpolation,
    considering only MTF values up to the point where it first drops below a threshold.

    Parameters
    ----------
    spatial_frequencies : np.array
        1D array of spatial frequency values (assumed to be sorted).
        mtf_values (np.array): 1D array of corresponding MTF values
        (assumed to be sorted according to spatial frequencies).
    cutoff_point : float
        The MTF value at which to find the cutoff frequency (e.g., 0.5 for 50%).
    mtf_threshold : float
        The minimum MTF value to consider for interpolation.
        Data points after the MTF first drops below this threshold
        will be excluded. Defaults to 0.01.

    Returns
    -------
    float or None
        The cutoff frequency at the specified MTF value, or None if the
        cutoff point is outside the relevant range of MTF values.
    """
    # Find the index of the first occurrence where MTF drops below the threshold
    below_threshold_index = np.argmax(mtf_values < mtf_threshold)

    # If MTF never drops below the threshold, use the entire array
    if below_threshold_index == 0 and not (mtf_values < mtf_threshold).any():
        valid_indices = np.arange(len(mtf_values))
    else:
        valid_indices = np.arange(
            below_threshold_index + 1
        )  # Include the point where it drops below

    if len(valid_indices) < 2:
        warnings.warn(
            f"The first MTF value is already below the threshold of {mtf_threshold}.",
            UserWarning,
        )
        return None

    # Use only the spatial frequencies and MTF values up to the threshold point
    valid_spatial_frequencies = spatial_frequencies[valid_indices]
    valid_mtf_values = mtf_values[valid_indices]

    # Create an interpolation function using the filtered data
    interpolation_function = interp1d(
        valid_mtf_values,
        valid_spatial_frequencies,
        bounds_error=False,
        fill_value="extrapolate",
    )

    # Find the cutoff frequency using the interpolation function
    cutoff_frequency = interpolation_function(cutoff_point)

    # Check if the cutoff point was within the filtered MTF range
    if (cutoff_point < np.min(valid_mtf_values)) or (
        cutoff_point > np.max(valid_mtf_values)
    ):
        cutoff_warning_message = (
            f"The cutoff point of {cutoff_point} is outside"
            + "the range of filtered MTF values (down to"
            + " first occurrence below {mtf_threshold}). "
            + "The returned cutoff frequency might be an "
            + "extrapolation based on the filtered data."
        )
        warnings.warn(
            cutoff_warning_message,
            UserWarning,
        )

    return cutoff_frequency


class Regressor(Enum):
    linear = LinearRegression
    ransac = RANSACRegressor
    huber = HuberRegressor


@dataclass
class ESF:
    x: np.array
    esf: np.array
    edge_angle: float = None  # angle in degrees
    regression_method: str = None


class MTF:
    def __init__(
        self, f: np.array, mtf: np.array, esf: ESF = None, lsf: np.array = None
    ) -> None:
        self._f = f
        self._mtf = mtf
        if esf:
            self.esf = esf.esf
            self.x = esf.x
            self.edge_angle = esf.edge_angle
            self.esf_regression_method = esf.regression_method
        else:
            self.esf = None
            self.x = None
            self.edge_angle = None
            self.esf_regression_method = None
        self.lsf = lsf
        self.num_positive_samples = int(len(self._mtf) / 2)

    @property
    def f(self):
        return self._f[: self.num_positive_samples]

    @property
    def f50(self):
        return get_cutoff_frequency(self.f, self.mtf, 0.5, 0.49)

    @property
    def f10(self):
        return get_cutoff_frequency(self.f, self.mtf, 0.1, 0.09)

    @property
    def mtf(self):
        return self._mtf[: self.num_positive_samples]

    def get_cutoff(self, cutoff_point, mtf_threshold: float = 0.09) -> float:
        return get_cutoff_frequency(self.f, self.mtf, cutoff_point, mtf_threshold)


@njit(parallel=True, fastmath=True)
def rebin_calc_esf(
    sample_positions: np.array, roi: np.ndarray, dists_upsampled: np.array
) -> np.array:
    """
    Rebin ROI pixel values according to their distance from the edge.
    Returns ESF.
    Uses numba for faster looping.
    """
    # Flatten arrays as numba doesn't like 'complex' indexing.
    roi = roi.flatten()
    dists_upsampled = dists_upsampled.flatten()
    n = len(sample_positions)
    esf_array = np.zeros(n)
    for i in prange(n):
        bin_val = sample_positions[i]
        inds = np.where(dists_upsampled == bin_val)[0]
        esf_array[i] = roi[inds].mean()

    return esf_array


def get_edge_coordinates(roi_canny: np.ndarray) -> np.array:
    """
    roi_canny: ROI output of detect_edge

    Returns N x 2 array of row, column indices of edge locations.
    If there are no gaps in the edge, N = roi_canny.shape[1], or the number of
    columns in roi_canny.
    """
    _, num_cols = roi_canny.shape

    edge_coords = []
    for i in np.arange(num_cols):
        try:
            yedge_pos = np.where(roi_canny[:, i] == roi_canny.max())[0][0]
            edge_coords.append([i, yedge_pos])
        except IndexError:  # If there are gaps in the edge.
            pass

    return np.array(edge_coords)


def get_edge_subpixel(
    detected_edge_coords: np.ndarray,
    regression_model: Regressor,
    edge_col_index: np.array,
) -> np.array:
    """
    detected_edge_coords: N x 2 array (row, column)
    regression_model: type of regressor, see Regressor class
    edge_col_index: column indices to interpolate the subpixel edges to

    edge_col_index may be different to detected_edge_coords[:, 0] if there are
    gaps in the edge. The regressor will therefore still try to fit a line to
    the edge as if the edge were continuous.
    """
    edge_col_index = edge_col_index.reshape(-1, 1)
    X = detected_edge_coords[:, 0].reshape(-1, 1)
    model = regression_model.value()
    model.fit(X, detected_edge_coords[:, 1])
    return model.predict(edge_col_index)


def get_esf(
    roi: np.ndarray,
    roi_canny: np.ndarray = None,
    edge_direction: str = "vertical",
    num_edge_samples: int = 2048,
    supersample_factor: int = 10,
    sample_period: float = 1,
    regressor: str = "huber",
) -> tuple[np.array, np.array]:
    """
    Get ESF from a ROI containing an edge.
    roi_canny is the edge-detected roi. If it is None, edge will be detected.

    Returns ESF values and x (sample position) values.
    """
    # Detect edge if detected edge roi not provided.
    if roi_canny is None:
        roi_canny = detect_edge_with_outlier_replacement(roi)

    if edge_direction == "horizontal":
        yn, xn = roi.shape
    elif edge_direction == "vertical":
        roi = roi.T
        roi_canny = roi_canny.T
        yn, xn = roi.shape

    edge_coords = get_edge_coordinates(roi_canny)
    edge_subpixel = get_edge_subpixel(edge_coords, Regressor[regressor], np.arange(xn))

    # Calculate angle of edge
    m = np.abs((edge_subpixel[-1] - edge_subpixel[0]) / xn)
    theta = np.degrees(np.arctan(m))
    if np.isclose(theta, 0):
        raise ESFAngleError("Edge angle found to be close to 0 degrees.")

    # Create an image where each pixel value is the vertical distance between
    # the pixel position and edge position for the pixels' respective columns.
    # Calculates distance from pixel centres.
    meshcol = np.repeat(np.arange(yn).reshape(-1, 1), xn, axis=1) + 0.5
    dists_from_edge = meshcol - edge_subpixel.reshape(1, -1)

    dists_upsampled = dists_from_edge * supersample_factor
    dists_upsampled = np.round(dists_upsampled) / supersample_factor

    sample_positions = np.linspace(
        -(num_edge_samples / 2 - 1) / supersample_factor,
        (num_edge_samples / 2) / supersample_factor + 1 / supersample_factor,
        num_edge_samples,
    )
    sample_positions = (
        np.round(supersample_factor * sample_positions) / supersample_factor
    )

    esf = rebin_calc_esf(sample_positions, roi, dists_upsampled)

    return ESF(sample_period * sample_positions, esf, theta, regressor)


def esf2mtf(esf: ESF) -> MTF:
    """
    Finite difference on esf and then FT.
    Returns MTF and frequencies.
    """
    lsf = np.convolve(esf.esf, [-1, 1], mode="valid")
    lsf = np.append([lsf[0]], lsf)  # Make the lsf same length as esf

    LSF = fft(lsf)
    mtf_array = np.abs(LSF) / np.abs(LSF).max()
    freqs = fftfreq(len(lsf), esf.x[1] - esf.x[0])
    return MTF(freqs, mtf_array, esf, lsf)


def monotone_esf(esf: ESF) -> np.array:
    """
    Applies monotonicity constraint to ESF to remove noise.
    """
    isoreg = IsotonicRegression(increasing="auto").fit(esf.x, esf.esf)
    esf_new_array = isoreg.predict(esf.x)
    esf_new = ESF(
        esf.x,
        esf_new_array,
        edge_angle=esf.edge_angle,
        regression_method=esf.regression_method,
    )
    return esf_new


def get_mtfs(
    dcm_path: str | Path,
    sample_period: float,
    esf_conditioning: bool = True,
) -> dict[tuple[np.array, np.array]]:
    """
    Reads image, performs edge detection and returns MTF for all edges of tool.
    """
    dcm = pydicom.dcmread(dcm_path)
    # The collimator is visible on the edge of Hologic images, remove.
    # For magnification images, remove the paddle.
    # For tomo, find in-focus slice.
    cropped = preprocess_dcm(dcm).array
    # Get the ROIs around the edges of the imaged MTF tool.
    rois, rois_canny = get_labelled_rois(cropped)

    mtfs = {}
    for edge_pos in rois:

        if edge_pos in ("left", "right"):
            edge_dir = "vertical"
        elif edge_pos in ("top", "bottom"):
            edge_dir = "horizontal"

        edge_roi = rois[edge_pos]
        edge_roi_canny = rois_canny[edge_pos]
        esf = get_esf(edge_roi, edge_roi_canny, edge_dir, sample_period=sample_period)
        if esf_conditioning:
            esf = monotone_esf(esf)  # Apply monotonicity constraint
        mtf = esf2mtf(esf)
        mtfs[edge_pos] = mtf

    return mtfs


def calculate_mtf(
    roi: np.ndarray,
    sample_period: float,
    roi_canny: np.ndarray = None,
    edge_dir: str = "vertical",
    esf_conditioning: bool = True,
) -> MTF:
    """
    Calculates MTF given ROI containing an edge.
    Returns frequencies and their corresponding MTF values as numpy arrays.
    """
    if roi_canny is None:
        roi_canny = detect_edge_with_outlier_replacement(roi)

    esf = get_esf(roi, roi_canny, edge_dir, sample_period=sample_period)
    if esf_conditioning:
        esf = monotone_esf(esf)  # Apply monotonicity constraint

    return esf2mtf(esf)
