"""
Algorithm based on Kao et al. (2005), 
"A Software tool for measurement of the modulation transfer function"
"""
import cv2
import pydicom
from scipy.fft import fft, fftfreq
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state


def rescale_pixels(img, new_max=255, new_min=0):
    """
    Rescale pixel values in image.
    """
    old_max, old_min = img.max(), img.min()
    new_img = (img-old_min)*((new_max - new_min)/(old_max - old_min)) + new_min
    return new_img


def calculate_distance(v1, v2):
    """
    Calculates Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((v1 - v2)**2))


def get_corner_pixels(canny_image):
    """
    Finds pixel values of the corners of the MTF edge.
    canny image is an image where edge detection has been performed.
    Labels output as most left, most right, most top and most bottom.
    """
    edge_idx = np.where(canny_image!=0)
    row_edge, col_edge = edge_idx
    
    # Find the corners of the edge rectangle
    left_idx = row_edge[np.where(col_edge==col_edge.min())][0]
    right_idx = row_edge[np.where(col_edge==col_edge.max())][-1]
    top_idx = col_edge[np.where(row_edge==row_edge.min())][0]
    bottom_idx = col_edge[np.where(row_edge==row_edge.max())][0]
    # Corners are labelled in terms of the most extreme position
    most_left = np.array((left_idx, col_edge.min()))
    most_right = np.array((right_idx, col_edge.max()))
    most_top = np.array((row_edge.min(), top_idx))
    most_bottom = np.array((row_edge.max(), bottom_idx))
    
    return most_left, most_right, most_top, most_bottom


def label_corners(most_left, most_right, most_top, most_bottom):
    """
    Returns a corners dictionary, with the pixel coordinates of top_left, 
    top_right, bottom_left, bottom_right corners.
    """
    corners = {}
    # Determine whether the MTF tool is tilted towards the left or right from the vertical image axis.    
    d1 = calculate_distance(most_left, most_top)
    d2 = calculate_distance(most_right, most_top)

    if d1 < d2:
        # Tilted towards the left
        corners['top_left'] = most_left
        corners['top_right'] = most_top
        corners['bottom_left'] = most_bottom
        corners['bottom_right'] = most_right
    elif d1 > d2:
        # Tilted towards the right
        corners['top_left'] = most_top
        corners['top_right'] = most_right
        corners['bottom_left'] = most_left
        corners['bottom_right'] = most_bottom

    return corners


def get_roi_bounds(corners):
    """
    Get row and columns bounds for ROIs around each edge of the tool.
    """
    # Get centre pixel coordinates of each edge
    left_centre = (corners['top_left'] + corners['bottom_left'])/2
    right_centre = (corners['top_right'] + corners['bottom_right'])/2
    top_centre = (corners['top_left'] + corners['top_right'])/2
    bottom_centre = (corners['bottom_left'] + corners['bottom_right'])/2
    
    tool_height = np.min(
        (calculate_distance(corners['top_left'], corners['bottom_left']),
        calculate_distance(corners['top_right'], corners['bottom_right']))
    )
    
    tool_width = np.min(
        (calculate_distance(corners['top_left'], corners['top_right']),
        calculate_distance(corners['bottom_left'], corners['bottom_right']))
    )
    
    # Define heights for vertical and horizontal edge rois
    height_ver = 0.9*tool_height
    height_hor = 0.9*tool_height
    
    # Define widths for vertical and horizontal edge rois
    width_ver = 1.8*tool_width
    width_hor = 0.9*tool_width

    roi_bounds = {
        'left': (
            (int(left_centre[0]-height_ver/2), int(left_centre[0]+height_ver/2)),
            (int(left_centre[1]-width_ver/2), int(left_centre[1]+width_ver/2))
        ),
        'right': (
            (int(right_centre[0]-height_ver/2), int(right_centre[0]+height_ver/2)),
            (int(right_centre[1]-width_ver/2), int(right_centre[1]+width_ver/2))
        ),
        'top': (
            (int(top_centre[0]-height_hor/2), int(top_centre[0]+height_hor/2)),
            (int(top_centre[1]-width_hor/2), int(top_centre[1]+width_hor/2))
        ),
        'bottom': (
            (int(bottom_centre[0]-height_hor/2), int(bottom_centre[0]+height_hor/2)),
            (int(bottom_centre[1]-width_hor/2), int(bottom_centre[1]+width_hor/2))
        )
    }

    return roi_bounds


def get_rois(image, roi_bounds):
    """
    Get rois from images defined by roi bounds.
    
    roi_bounds is a dictionary containing keys and ((x1, x2), (y1, y2)) where
    x1/x2 are the row bounds and y1/y2 are the columns bounds.
    """
    rois = {}
    for key, val in roi_bounds.items():
        row_vals, col_vals = val
        rois[key] = image[row_vals[0]:row_vals[1], col_vals[0]:col_vals[1]]
    return rois


def get_esf(roi, roi_canny=None, edge_direction='vertical', 
    num_edge_samples=2048, supersample_factor=10):
    """
    Get ESF from a ROI containing an edge.
    roi_canny is the edge-detected roi. If it is None, edge will be detected.
    
    Returns ESF values and x (sample position) values.
    """
    # Detect edge if detected edge roi not provided.
    if roi_canny is None:
        roi_canny = rescale_pixels(roi).astype(np.uint8)
        roi_canny = cv2.Canny(roi_canny, 100, 200)

    if edge_direction == 'vertical':
        xn, yn = roi.shape
    elif edge_direction == 'horizontal':
        roi = roi.T
        roi_canny = roi_canny.T
        xn, yn = roi.shape

    edge_coords = []
    for i in np.arange(xn):
        yedge_pos = np.where(roi_canny[i, :]==roi_canny.max())[0][0]
        edge_coords.append([i, yedge_pos])
    
    edge_coords = np.array(edge_coords)
    
    m, b = np.polyfit(edge_coords[:,0], edge_coords[:,1], 1)
    
    # Edge location subpixel
    x = np.arange(xn)
    y = m*x + b
    
    # Create an image where each pixel value is the horizontal distance between 
    # the pixel position and edge position for the pixels' respective rows.
    # Calculates distance from pixel centres.
    meshrow = np.repeat(np.arange(yn).reshape(1,-1), xn, axis=0) + 0.5
    dists_horizontal = meshrow - y.reshape(-1, 1)
    
    dists_upsampled = dists_horizontal*supersample_factor
    dists_upsampled = np.round(dists_upsampled)/supersample_factor
    
    # Re-bin pixel values based on their distance from the edge
    unique_bins = np.unique(dists_upsampled)
    
    sample_positions = np.linspace(-(num_edge_samples/2-1)/supersample_factor, 
                                   (num_edge_samples/2)/supersample_factor+1/supersample_factor, num_edge_samples)
    
    esf = np.zeros(num_edge_samples)
    
    for i, bin_val in enumerate(sample_positions):
        esf[i] = roi[dists_upsampled==np.round(supersample_factor*bin_val)/supersample_factor].mean()
        
    return esf, sample_positions


def esf2mtf(esf, sample_period):
    """
    Finite difference on esf and then FT.
    Returns MTF and frequencies.
    """
    lsf = np.convolve(esf, [-1, 1], mode='valid')
    lsf = np.append([lsf[0]], lsf) # Make the lsf same length as esf
    
    LSF = fft(lsf)
    MTF = np.abs(LSF)/np.abs(LSF).max()
    freqs = fftfreq(len(lsf), sample_period)
    return MTF, freqs


def monotone_esf(esf, sample_positions):
    """
    Applies monotonicity constraint to ESF to remove noise.
    """
    isoreg = IsotonicRegression(increasing='auto').fit(sample_positions, esf)
    esf_new = isoreg.predict(sample_positions)
    return esf_new


def get_mtfs(dcm_path, sample_period):
    """
    Reads image, performs edge detection and returns MTF for all edges of tool.
    """
    dcm = pydicom.dcmread(dcm_path)
    # The collimator is visible on the edge of Hologic images, remove.
    # Rescale pixel values for conversion to 8 bit
    img = rescale_pixels(dcm.pixel_array[12:-12, :])
    img8bit = img.astype(np.uint8)
    # Perform edge detection
    imgedge = cv2.Canny(img8bit, 100, 200)
    # Get tool edge corners and label them.
    most_left, most_right, most_top, most_bottom = get_corner_pixels(imgedge)
    corners = label_corners( most_left, most_right, most_top, most_bottom)
    # Get ROIs
    roi_bounds = get_roi_bounds(corners)
    rois = get_rois(img, roi_bounds)
    rois_canny = get_rois(imgedge, roi_bounds)

    mtfs = {}
    for edge_pos in ('left', 'right', 'top', 'bottom'):

        if edge_pos in ('left', 'right'):
            edge_dir = 'vertical'
        elif edge_pos in ('top', 'bottom'):
            edge_dir = 'horizontal'

        edge_roi = rois[edge_pos]
        edge_roi_canny = rois_canny[edge_pos]
        esf, sample_positions = get_esf(edge_roi, edge_roi_canny, edge_dir, 10)
        MTF, freqs = esf2mtf(esf, sample_period/10)

        mtfs[edge_pos] = (freqs, MTF)

    return mtfs

    
        
