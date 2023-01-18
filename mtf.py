"""
Algorithm based on Kao et al. (2005), 
"A Software tool for measurement of the modulation transfer function"

Drawing ROIs around the edges of the MTF tool is automated by the get_labelled_rois
function.
"""
from pathlib import Path
import pydicom
from scipy.fft import fft, fftfreq
from numba import njit, prange
import numpy as np
from sklearn.isotonic import IsotonicRegression
from .roifind import get_labelled_rois, detect_edge
from .dcmutils import preprocess_dcm


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
    esf = np.zeros(n)
    for i in prange(n):
        bin_val = sample_positions[i]
        inds = np.where(dists_upsampled == bin_val)[0]
        esf[i] = roi[inds].mean()

    return esf


def get_edge_coordinates(
    roi_canny: np.ndarray, num_edge_samples: int, supersample_factor: int
) -> np.array:
    num_rows, num_cols = roi_canny.shape
    num_pixel_samples = np.ceil(num_edge_samples / supersample_factor)

    col_mid = int(num_cols / 2)
    edge_find_lower = col_mid - int(num_pixel_samples / 2)
    edge_find_upper = col_mid + int(num_pixel_samples / 2)
    edge_coords = []
    for i in np.arange(num_rows):
        yedge_pos = np.where(
            roi_canny[i, edge_find_lower:edge_find_upper] == roi_canny.max()
        )[0][0]
        yedge_pos += edge_find_lower
        edge_coords.append([i, yedge_pos])

    return np.array(edge_coords)


def get_esf(
    roi: np.ndarray,
    roi_canny: np.ndarray = None,
    edge_direction: str = "vertical",
    num_edge_samples: int = 2048,
    supersample_factor: int = 10,
) -> tuple[np.array, np.array]:
    """
    Get ESF from a ROI containing an edge.
    roi_canny is the edge-detected roi. If it is None, edge will be detected.

    Returns ESF values and x (sample position) values.
    """
    # Detect edge if detected edge roi not provided.
    if roi_canny is None:
        roi_canny = detect_edge(roi)

    if edge_direction == "vertical":
        xn, yn = roi.shape
    elif edge_direction == "horizontal":
        roi = roi.T
        roi_canny = roi_canny.T
        xn, yn = roi.shape

    edge_coords = get_edge_coordinates(roi_canny, num_edge_samples, supersample_factor)

    m, b = np.polyfit(edge_coords[:, 0], edge_coords[:, 1], 1)

    # Edge location subpixel
    x = np.arange(xn)
    y = m * x + b

    # Create an image where each pixel value is the horizontal distance between
    # the pixel position and edge position for the pixels' respective rows.
    # Calculates distance from pixel centres.
    meshrow = np.repeat(np.arange(yn).reshape(1, -1), xn, axis=0) + 0.5
    dists_horizontal = meshrow - y.reshape(-1, 1)

    dists_upsampled = dists_horizontal * supersample_factor
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

    return esf, sample_positions


def esf2mtf(esf: np.array, sample_period: float) -> tuple[np.array, np.array]:
    """
    Finite difference on esf and then FT.
    Returns MTF and frequencies.
    """
    lsf = np.convolve(esf, [-1, 1], mode="valid")
    lsf = np.append([lsf[0]], lsf)  # Make the lsf same length as esf

    LSF = fft(lsf)
    MTF = np.abs(LSF) / np.abs(LSF).max()
    freqs = fftfreq(len(lsf), sample_period)
    return MTF, freqs


def monotone_esf(esf: np.array, sample_positions: np.array) -> np.array:
    """
    Applies monotonicity constraint to ESF to remove noise.
    """
    isoreg = IsotonicRegression(increasing="auto").fit(sample_positions, esf)
    esf_new = isoreg.predict(sample_positions)
    return esf_new


def get_mtfs(
    dcm_path: str | Path, sample_period: float
) -> dict[tuple[np.array, np.array]]:
    """
    Reads image, performs edge detection and returns MTF for all edges of tool.
    """
    dcm = pydicom.dcmread(dcm_path)
    # The collimator is visible on the edge of Hologic images, remove.
    # For magnification images, remove the paddle.
    # For tomo, find in-focus slice.
    cropped = preprocess_dcm(dcm)
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
        esf, sample_positions = get_esf(edge_roi, edge_roi_canny, edge_dir)
        esf = monotone_esf(esf, sample_positions)  # Apply monotonicity constraint
        MTF, freqs = esf2mtf(esf, sample_period / 10)

        mtfs[edge_pos] = (freqs, MTF)

    return mtfs


def calculate_mtf(
    roi: np.ndarray,
    sample_period: float,
    roi_canny: np.ndarray = None,
    edge_dir: str = "vertical",
    sample_number: int = None,
) -> tuple[np.array, np.array]:
    """
    Calculates MTF given ROI containing an edge.
    """
    if roi_canny is None:
        roi_canny = detect_edge(roi)

    esf, sample_positions = get_esf(roi, roi_canny, edge_dir)
    esf = monotone_esf(esf, sample_positions)  # Apply monotonicity constraint
    MTF, freqs = esf2mtf(esf, sample_period / 10)

    if not sample_number:
        sample_number = int(len(MTF) / 2)

    return freqs[:sample_number], MTF[:sample_number]
