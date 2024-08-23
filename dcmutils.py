from pydicom.dataset import FileDataset
import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MammoMTFImage:
    """Container class for preprocessed mammo MTF edge images."""

    array: np.ndarray  # 2D pixel array
    acquisition: str = "conventional"  # "conventional", "tomo", "mag"
    manufacturer: str = None
    orientation: str = "left"  # "left" means that the chest wall is on the right
    focus_plane: str = None  # for tomo, the slice number corresponding to 2D array


def preprocess_dcm(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    """
    Preprocesses DICOM image, returning a pixel array for MTF calculation.
    """
    manufacturer_name = dcm[0x0008, 0x0070].value.lower()
    if "hologic" in manufacturer_name:
        return preprocess_hologic(dcm, acquisition=acquisition)
    elif "fuji" in manufacturer_name:
        return preprocess_fuji(dcm, acquisition=acquisition)
    elif "ge" in manufacturer_name:
        return preprocess_ge(dcm, acquisition=acquisition)
    else:
        raise ValueError(f"Image from unsupported manufacturer: {manufacturer_name}")


def autofocus_tomo(
    tomo_recon: np.ndarray, manufacturer: str = None, orientation: str = None
) -> np.ndarray:
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
    array2d = tomo_recon[max_lapvar_slice, ...]
    return MammoMTFImage(array2d, "tomo", manufacturer, orientation, max_lapvar_slice)


def preprocess_hologic(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    arr = dcm.pixel_array
    if not acquisition:
        img_type_header = dcm[0x0008, 0x0008].value
        if "TOMOSYNTHESIS" in img_type_header or "VOLUME" in img_type_header:
            mtf_image = autofocus_tomo(arr, manufacturer="hologic", orientation="left")
        else:
            # Get value for (0018, 11a4) Paddle description
            paddleval = dcm[0x0018, 0x11A4].value
            if paddleval == "10CM MAG":
                mtf_image = _preprocess_hologic_mag(arr)
            else:
                mtf_image = _preprocess_hologic_conventional(arr)
    elif acquisition == "conventional":
        mtf_image = _preprocess_hologic_conventional(arr)
    elif acquisition == "mag":
        mtf_image = _preprocess_hologic_mag(arr)
    elif acquisition == "tomo":
        mtf_image = _preprocess_hologic_tomo(arr)
    else:
        raise ValueError(
            "acquisition should be either None, 'conventional', 'mag' or 'tomo'"
        )

    return mtf_image


def _preprocess_hologic_conventional(pixel_array: np.ndarray) -> MammoMTFImage:
    rowlims = (20, -20)
    pixel_array = pixel_array[rowlims[0] : rowlims[1], :]
    return MammoMTFImage(
        pixel_array, acquisition="conventional", manufacturer="hologic"
    )


def _preprocess_hologic_mag(pixel_array: np.ndarray) -> MammoMTFImage:
    rowlims = (450, 2800)
    pixel_array = pixel_array[rowlims[0] : rowlims[1], :]
    return MammoMTFImage(pixel_array, acquisition="mag", manufacturer="hologic")


def _preprocess_hologic_tomo(pixel_array: np.ndarray) -> MammoMTFImage:
    return autofocus_tomo(pixel_array, manufacturer="hologic", orientation="left")


def linearise_fuji(pixel_array: np.ndarray) -> np.ndarray:
    lin = 10 ** ((pixel_array / 4 - 2047) / 1024)
    rescaled = lin * 2**12 / lin.max()
    return rescaled.astype(np.uint16)


def preprocess_fuji(dcm: FileDataset, acquisition: str = None) -> np.ndarray:
    arr = dcm.pixel_array
    if not acquisition:
        img_type_header = dcm[0x0008, 0x0008].value
        if "TOMOSYNTHESIS" in img_type_header or "VOLUME" in img_type_header:
            mtf_image = autofocus_tomo(arr, manufacturer="fuji", orientation="right")
        else:
            arr = linearise_fuji(arr)
            paddlevel = dcm[0x0018, 0x11A4].value
            if "MAG" in paddlevel:
                mtf_image = MammoMTFImage(
                    arr, acquisition="mag", manufacturer="fuji", orientation="right"
                )
            else:
                mtf_image = MammoMTFImage(
                    arr,
                    acquisition="conventional",
                    manufacturer="fuji",
                    orientation="right",
                )
    return mtf_image


def preprocess_ge(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    arr = dcm.pixel_array
    return MammoMTFImage(
        arr, manufacturer="ge", acquisition="conventional", orientation="right"
    )
