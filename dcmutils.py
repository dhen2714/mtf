from pydicom.dataset import FileDataset
import cv2
import numpy as np


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
