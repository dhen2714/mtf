"""
Functions to automatically draw regions of interest (ROIs) around the edges of
the MTF edge tool in mammography.

1) Performs Canny edge detection on entire image.
2) Finds the contours in the edge detected image.
3) Draws a bounding box around around the biggest contour, assumed to be the
edge tool.
4) Find the midpoints of the sides of the bounding box, from there finding the
midpoints of each edge in pixel coordinates.
5) Using the midpoints of each edge as the centre, define ROIs around the edge,
with dimensions that depend on the size of the bounding box.
6) The ROIs are labelled as 'left', 'right', 'top' or 'bottom'.

The function get_labelled_rois returns two dictionaries, one with labelled ROIs
extracted from the original image, and one with labelled ROIs from the edge-
detected image. The edge-detected image is used to find the ESF.
"""
import cv2
import numpy as np


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


def detect_edge(image_array: np.ndarray) -> np.ndarray:
    image8bit = rescale_pixels(image_array).astype(np.uint8)
    return cv2.Canny(image8bit, 100, 300)


def get_labelled_rois(image: np.ndarray) -> tuple[dict, dict]:
    """
    Returns dictionary of labelled rois.
    """
    roi_bounds = get_roi_bounds(image)
    roi_bounds = fix_rois(roi_bounds, *image.shape)
    rois = get_rois(image, roi_bounds)
    rois_canny = {}
    for roi_name, roi in rois.items():
        rois_canny[roi_name] = detect_edge(roi)
    return rois, rois_canny


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
    """
    Searches for edge midpoint pixel along a 1D slice that includes the bounding box contour.
    The edge midpoint will be the midpoint of the ROI for which the ESF is calculated.
    """
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


def get_roi_bounds(image: np.ndarray) -> dict[tuple[tuple[int, int], tuple[int, int]]]:
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
    # Filter image to remove dead pixels, lines that may affect edge detection
    image_filtered = cv2.medianBlur(image, 5)
    edges_filtered = detect_edge(image_filtered)
    x, y, w, h = bound_edge_tool(edges_filtered)
    # Get midpoint for each side
    contour_midpoints = {
        "left": np.array([y, x]) + np.array([int((h - 1) / 2), 0]),
        "right": np.array([y, x]) + np.array([int((h - 1) / 2), (w - 1)]),
        "top": np.array([y, x]) + np.array([0, int((w - 1) / 2)]),
        "bottom": np.array([y, x]) + np.array([(h - 1), int((w - 1) / 2)]),
    }
    contour_midpoints = {
        "left": np.array([int((h - 1) / 2), 0]),
        "right": np.array([int((h - 1) / 2), (w - 1)]),
        "top": np.array([0, int((w - 1) / 2)]),
        "bottom": np.array([(h - 1), int((w - 1) / 2)]),
    }
    bounded_area = image[y : y + h, x : x + w]
    canny = detect_edge(bounded_area)
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
                y + int(roi_midpoint[0] - roi_height / 2),
                y + int(roi_midpoint[0] + roi_height / 2),
            ),
            (
                x + int(roi_midpoint[1] - roi_width / 2),
                x + int(roi_midpoint[1] + roi_width / 2),
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
    size_check = (row_length > 205) and (col_length > 205)
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
