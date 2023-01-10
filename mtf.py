"""
Algorithm based on Kao et al. (2005), 
"A Software tool for measurement of the modulation transfer function"
"""
from pathlib import Path
import cv2
import pydicom
from pydicom.dataset import FileDataset
from scipy.fft import fft, fftfreq
import numpy as np
from sklearn.isotonic import IsotonicRegression


def rescale_pixels(
    image: np.ndarray, new_max: float = 255, new_min: float = 0
) -> np.ndarray:
    """
    Rescale pixel values in image.
    """
    old_max, old_min = image.max(), image.min()
    new_image = (image - old_min) * (
        (new_max - new_min) / (old_max - old_min)
    ) + new_min
    return new_image


def calculate_distance(v1: np.array, v2: np.array) -> float:
    """
    Calculates Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((v1 - v2) ** 2))


def bound_edge_tool(canny: np.ndarray) -> tuple[int, int, int, int]:
    """
    Return x, y, w, h, where x, y are the column, row indices of the top left
    corner of the bounding box around edge tool. w and h are the width and
    height of the box, respectively
    """
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_len = 0
    # Assumes the biggest contour is the MTF tool.
    for i, c in enumerate(contours):
        contour_length = len(c)
        if contour_length > max_len:
            max_len = contour_length
            contour_ind = i

    mtfedge = contours[contour_ind]
    # Define rectangle around the MTF edge.
    # rect returns x,y,w,h where x,y are the column, row indices of top left corner
    # w, h are the number of columns and rows, respectively
    return cv2.boundingRect(mtfedge)


def contourmid2roimid(
    contour_midpoint: np.array,
    canny: np.ndarray,
    edge_location: str,
    search_length: int,
) -> np.array:
    edge_slices = {
        "left": canny[
            contour_midpoint[0],
            contour_midpoint[1] : contour_midpoint[1] + search_length,
        ],
        "right": canny[
            contour_midpoint[0],
            contour_midpoint[1] - search_length : contour_midpoint[1],
        ],
        "top": canny[
            contour_midpoint[0] : contour_midpoint[0] + search_length,
            contour_midpoint[1],
        ],
        "bottom": canny[
            contour_midpoint[0] - search_length : contour_midpoint[0],
            contour_midpoint[1],
        ],
    }
    slice_start = {
        "left": np.array([contour_midpoint[0], contour_midpoint[1]]),
        "right": np.array([contour_midpoint[0], contour_midpoint[1] - search_length]),
        "top": np.array([contour_midpoint[0], contour_midpoint[1]]),
        "bottom": np.array([contour_midpoint[0] - search_length, contour_midpoint[1]]),
    }
    search_slice = edge_slices[edge_location]
    local_idx = np.where(search_slice != 0)[0]
    if local_idx.size > 0:
        if edge_location in ["left", "right"]:
            roi_midpoint = slice_start[edge_location] + np.array([0, local_idx[0]])
        elif edge_location in ["top", "bottom"]:
            roi_midpoint = slice_start[edge_location] + np.array([local_idx[0], 0])
    else:
        roi_midpoint = None

    return roi_midpoint


def get_roi_bounds(canny: np.ndarray) -> dict[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Get the indicies for the row and column bounds for edge ROIs.

    Returns a dictionary with index entries for each edge. Each entry is a
    a tuple of tuples, with the first tuple being the first and last row
    indices for the ROI, and the second tuple being the first and last
    column indices for the ROI. For example:
    {
        'left': ((row_start, row_end), (col_start, col_end)),
        'right': ((row_start, row_end), (col_start, col_end))
    }
    """
    x, y, w, h = bound_edge_tool(canny)
    # Get midpoint for each side
    contour_midpoints = {
        "left": np.array([y, x]) + np.array([int((h - 1) / 2), 0]),
        "right": np.array([y, x]) + np.array([int((h - 1) / 2), (w - 1)]),
        "top": np.array([y, x]) + np.array([0, int((w - 1) / 2)]),
        "bottom": np.array([y, x]) + np.array([(h - 1), int((w - 1) / 2)]),
    }
    # If the edge is not found within search_length, assumes no edge.
    search_length = 200

    # Define heights for vertical and horizontal edge rois
    height_ver = 0.7 * h
    height_hor = 0.8 * h

    # Define widths for vertical and horizontal edge rois
    width_ver = 1.6 * w
    width_hor = 0.6 * w

    roi_bounds = {}
    # Search for edge tool midpoints from contour midpoints.
    for key, val in contour_midpoints.items():
        roi_midpoint = contourmid2roimid(val, canny, key, search_length)
        if roi_midpoint is None:
            continue
        if key in ["left", "right"]:
            roi_height, roi_width = height_ver, width_ver
        elif key in ["top", "bottom"]:
            roi_height, roi_width = height_hor, width_hor

        roi_bounds[key] = (
            (
                int(roi_midpoint[0] - roi_height / 2),
                int(roi_midpoint[0] + roi_height / 2),
            ),
            (
                int(roi_midpoint[1] - roi_width / 2),
                int(roi_midpoint[1] + roi_width / 2),
            ),
        )

    return roi_bounds


def get_rois(
    image: np.ndarray, roi_bounds: dict[tuple[tuple[int, int], tuple[int, int]]]
) -> dict[np.ndarray]:
    """
    Get rois from images defined by roi bounds.

    roi_bounds is a dictionary containing keys and ((x1, x2), (y1, y2)) where
    x1/x2 are the row bounds and y1/y2 are the columns bounds.
    """
    rois = {}
    for key, val in roi_bounds.items():
        row_vals, col_vals = val
        rois[key] = image[row_vals[0] : row_vals[1], col_vals[0] : col_vals[1]]
    return rois


def check_roi_size(row_bounds: tuple[int, int], col_bounds: tuple[int, int]) -> bool:
    """
    Returns True if the size of the ROI is greater than 100x100 pixels.
    """
    row_length = row_bounds[1] - row_bounds[0]
    col_length = col_bounds[1] - col_bounds[0]
    size_check = (row_length > 100) and (col_length > 100)
    return size_check


def fix_rois(
    roi_bounds: tuple[tuple[int, int], tuple[int, int]], row_lim: int, col_lim: int
) -> dict[tuple[tuple[int, int], tuple[int, int]]]:
    """
    Resize ROIS to be symmetrical if they are too close to the edge of
    the image.
    """
    fixed_rois = {}
    for roi_name, roi in roi_bounds.items():
        row_bounds, col_bounds = roi
        new_row_bounds = row_bounds
        new_col_bounds = col_bounds

        if row_bounds[0] < 0:
            new_row_bounds = (0, row_bounds[1] + row_bounds[0])
        if row_bounds[1] >= row_lim:
            diff = row_bounds[1] - row_lim
            new_row_bounds = (row_bounds[0] + diff, row_lim - 1)

        if col_bounds[0] < 0:
            new_col_bounds = (0, col_bounds[1] + col_bounds[0])
        if col_bounds[1] >= col_lim:
            diff = col_bounds[1] - col_lim
            new_col_bounds = (col_bounds[0] + diff, col_lim - 1)

        new_bounds = (new_row_bounds, new_col_bounds)
        if check_roi_size(new_row_bounds, new_col_bounds):
            fixed_rois[roi_name] = new_bounds
    return fixed_rois


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
        roi_canny = rescale_pixels(roi).astype(np.uint8)
        roi_canny = cv2.Canny(roi_canny, 100, 300)

    if edge_direction == "vertical":
        xn, yn = roi.shape
    elif edge_direction == "horizontal":
        roi = roi.T
        roi_canny = roi_canny.T
        xn, yn = roi.shape

    edge_coords = []
    for i in np.arange(xn):
        yedge_pos = np.where(roi_canny[i, :] == roi_canny.max())[0][0]
        edge_coords.append([i, yedge_pos])

    edge_coords = np.array(edge_coords)

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

    esf = np.zeros(num_edge_samples)

    for i, bin_val in enumerate(sample_positions):
        esf[i] = roi[
            dists_upsampled
            == np.round(supersample_factor * bin_val) / supersample_factor
        ].mean()

    return esf, sample_positions


def esf2mtf(esf, sample_period):
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


def monotone_esf(esf, sample_positions):
    """
    Applies monotonicity constraint to ESF to remove noise.
    """
    isoreg = IsotonicRegression(increasing="auto").fit(sample_positions, esf)
    esf_new = isoreg.predict(sample_positions)
    return esf_new


def preprocess_dcm(dcm: FileDataset) -> np.ndarray:
    """
    Preprocesses DICOM image, returning a pixel array for MTF calculation.
    """
    manufacturer_name = dcm[0x0008, 0x0070].value.lower()
    if "hologic" in manufacturer_name:
        arr = preprocess_hologic(dcm)
    elif "ge" in manufacturer_name:
        arr = preprocess_ge(dcm)
    return arr


def autofocus_tomo(tomo_recon):
    """
    Find the tomosynthesis slice in which the MTF edge is in focus.
    Uses variance of Laplacian as a metric for focus.
    Input: 3D tomosynthesis reconstruction pixel array.
    Output: 2D pixel array representation of in-focus slice.
    """
    max_lapvar_slice = 0
    max_lapvar = 0
    for i, tomo_slice in enumerate(tomo_recon):
        lapvar = cv2.Laplacian(tomo_slice, cv2.CV_64F).var()
        if lapvar > max_lapvar:
            max_lapvar_slice = i
            max_lapvar = lapvar
    return tomo_recon[max_lapvar_slice, ...]


def preprocess_hologic(dcm):
    img_type_header = dcm[0x0008, 0x0008].value
    if "TOMOSYNTHESIS" in img_type_header:
        arr = autofocus_tomo(dcm.pixel_array)
    else:
        # Get value for (0018, 11a4) Paddle description
        paddleval = dcm[0x0018, 0x11A4].value
        arr = dcm.pixel_array
        if paddleval == "10CM MAG":
            rowlims = (450, 2800)
        else:
            rowlims = (20, -20)
        arr = arr[rowlims[0] : rowlims[1], :]
    return arr


def preprocess_ge(dcm):
    return dcm.pixel_array


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
    # Rescale pixel values for conversion to 8 bit
    img = rescale_pixels(cropped)
    img8bit = img.astype(np.uint8)
    # Perform edge detection
    imgedge = cv2.Canny(img8bit, 100, 300)
    # Get ROIs
    roi_bounds = get_roi_bounds(imgedge)
    roi_bounds = fix_rois(roi_bounds, *img.shape)
    rois = get_rois(img, roi_bounds)
    rois_canny = get_rois(imgedge, roi_bounds)

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


def get_labelled_rois(image: np.ndarray) -> tuple[dict, dict]:
    """
    Returns dictionary of labelled rois.
    """
    image = rescale_pixels(image)
    image_edge = cv2.Canny(image.astype(np.uint8), 100, 300)
    # Get ROIs
    roi_bounds = get_roi_bounds(image_edge)
    roi_bounds = fix_rois(roi_bounds, *image.shape)
    rois = get_rois(image, roi_bounds)
    rois_canny = get_rois(image_edge, roi_bounds)
    return rois, rois_canny


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
        roi_canny = rescale_pixels(roi).astype(np.uint8)
        roi_canny = cv2.Canny(roi_canny, 100, 300)

    esf, sample_positions = get_esf(roi, roi_canny, edge_dir)
    esf = monotone_esf(esf, sample_positions)  # Apply monotonicity constraint
    MTF, freqs = esf2mtf(esf, sample_period / 10)

    if not sample_number:
        sample_number = int(len(MTF) / 2)

    return freqs[:sample_number], MTF[:sample_number]
