from pydicom.dataset import FileDataset
import cv2
import numpy as np
from dataclasses import dataclass

try:
    from tomodd import decompress

    # https://github.com/dhen2714/tomo-decompressor
    print("Using tomodd decompressor")
    TOMODD_AVAILABLE = True
except ImportError:
    TOMODD_AVAILABLE = False


def get_tomo_array(dcm: FileDataset) -> np.ndarray:
    if TOMODD_AVAILABLE:
        arr = decompress(str(dcm.filename))
    else:
        arr = dcm.pixel_array
    return arr


@dataclass
class MammoMTFImage:
    """Container class for preprocessed mammo MTF edge images."""

    array: np.ndarray  # 2D pixel array, or 'focus plane' image for tomo
    acquisition: str = "conventional"  # "conventional", "tomo", "mag"
    manufacturer: str = None
    orientation: str = "left"  # "left" means chest wall is on the right of DICOM image
    pixel_spacing: float = None
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
    elif "siemens" in manufacturer_name:
        return preprocess_siemens(dcm, acquisition=acquisition)
    elif "ge" in manufacturer_name:
        return preprocess_ge(dcm, acquisition=acquisition)
    else:
        raise ValueError(f"Image from unsupported manufacturer: {manufacturer_name}")


def autofocus_tomo(
    tomo_recon: np.ndarray,
    manufacturer: str = None,
    orientation: str = None,
    pixel_spacing: float = None,
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
    return MammoMTFImage(
        array2d, "tomo", manufacturer, orientation, pixel_spacing, max_lapvar_slice
    )


def get_pixel_spacing(dcm: FileDataset) -> float:
    tagval = dcm.get((0x0018, 0x1164), None)
    try:
        tagval = float(tagval[0])
    except Exception:
        tagval = None
    return tagval


def get_orientation(dcm: FileDataset) -> str:
    tagval = dcm.get((0x0020, 0x0020), None)
    # tagval is something like [P, R] or [A, L].
    orientation_val = tagval[1]
    orientation_dict = {"L": "left", "R": "right"}
    return orientation_dict[orientation_val]


def get_pixel_spacing_tomo(dcm: FileDataset) -> float:
    try:
        dcm_slice_sequence = dcm[(0x5200, 0x9230)]
        dcm_element = dcm_slice_sequence[0][(0x0028, 0x9110)]
        pixel_spacing = float(dcm_element[0][(0x0028, 0x0030)].value[0])
    except Exception:
        pixel_spacing = None
    return pixel_spacing


def preprocess_hologic(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    if not acquisition:
        img_type_header = dcm[0x0008, 0x0008].value
        if "TOMOSYNTHESIS" in img_type_header or "VOLUME" in img_type_header:
            arr = get_tomo_array(dcm)
            pixel_spacing = get_pixel_spacing_tomo(dcm)
            mtf_image = autofocus_tomo(
                arr,
                manufacturer="hologic",
                orientation="left",
                pixel_spacing=pixel_spacing,
            )
        else:
            arr = dcm.pixel_array
            # Get value for (0018, 11a4) Paddle description
            paddleval = dcm[0x0018, 0x11A4].value
            pixel_spacing = get_pixel_spacing(dcm)
            if paddleval == "10CM MAG":
                mtf_image = _preprocess_hologic_mag(arr, pixel_spacing)
            else:
                mtf_image = _preprocess_hologic_conventional(arr, pixel_spacing)
    elif acquisition == "conventional":
        arr = dcm.pixel_array
        pixel_spacing = get_pixel_spacing(dcm)
        mtf_image = _preprocess_hologic_conventional(arr, pixel_spacing)
    elif acquisition == "mag":
        arr = dcm.pixel_array
        pixel_spacing = get_pixel_spacing(dcm)
        mtf_image = _preprocess_hologic_mag(arr, pixel_spacing)
    elif acquisition == "tomo":
        arr = get_tomo_array(dcm)
        pixel_spacing = get_pixel_spacing_tomo(dcm)
        mtf_image = _preprocess_hologic_tomo(arr, pixel_spacing)
    else:
        raise ValueError(
            "acquisition should be either None, 'conventional', 'mag' or 'tomo'"
        )

    return mtf_image


def _preprocess_hologic_conventional(
    pixel_array: np.ndarray, pixel_spacing: float = None
) -> MammoMTFImage:
    rowlims = (20, -20)
    pixel_array = pixel_array[rowlims[0] : rowlims[1], :]
    # For edge cases where there are still saturated pixels that haven't been clipped.
    max_pix_val = 2**14 - 1
    med_pix_val = np.median(pixel_array)
    satured_pixels = pixel_array == max_pix_val
    pixel_array[satured_pixels] = med_pix_val

    return MammoMTFImage(
        pixel_array,
        acquisition="conventional",
        manufacturer="hologic",
        pixel_spacing=pixel_spacing,
    )


def _preprocess_hologic_mag(
    pixel_array: np.ndarray, pixel_spacing: float = None
) -> MammoMTFImage:
    rowlims = (450, 2800)
    pixel_array = pixel_array[rowlims[0] : rowlims[1], :]
    return MammoMTFImage(
        pixel_array,
        acquisition="mag",
        manufacturer="hologic",
        pixel_spacing=pixel_spacing,
    )


def _preprocess_hologic_tomo(
    pixel_array: np.ndarray, pixel_spacing: float = None
) -> MammoMTFImage:
    return autofocus_tomo(
        pixel_array,
        manufacturer="hologic",
        orientation="left",
        pixel_spacing=pixel_spacing,
    )


def preprocess_siemens(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    arr = dcm.pixel_array
    orientation = get_orientation(dcm)
    if not acquisition:
        # Get value for (0018, 11a4) Detector Mode
        detector_mode = dcm[0x0018, 0x7008].value
        pixel_spacing = get_pixel_spacing(dcm)
        if detector_mode == "MAGNIFICATION":
            mtf_image = _preprocess_siemens_mag(arr, pixel_spacing, orientation)
        else:
            mtf_image = _preprocess_siemens_conventional(
                arr, pixel_spacing, orientation
            )
    elif acquisition == "conventional":
        pixel_spacing = get_pixel_spacing(dcm)
        mtf_image = _preprocess_siemens_conventional(arr, pixel_spacing, orientation)
    elif acquisition == "mag":
        pixel_spacing = get_pixel_spacing(dcm)
        mtf_image = _preprocess_siemens_mag(arr, pixel_spacing, orientation)
    else:
        raise ValueError(
            "Siemens acquisition should be either None, 'conventional' or 'mag'"
        )

    return mtf_image


def _preprocess_siemens_conventional(
    pixel_array: np.ndarray, pixel_spacing: float, orientation: str
) -> MammoMTFImage:
    rowlims = (100, -100)
    pixel_array = pixel_array[rowlims[0] : rowlims[1], :]
    return MammoMTFImage(
        pixel_array,
        acquisition="conventional",
        manufacturer="siemens",
        pixel_spacing=pixel_spacing,
        orientation=orientation,
    )


def _preprocess_siemens_mag(
    arr: np.ndarray, pixel_spacing: float, orientation: str
) -> MammoMTFImage:
    return MammoMTFImage(
        arr,
        acquisition="mag",
        manufacturer="siemens",
        pixel_spacing=pixel_spacing,
        orientation=orientation,
    )


def linearise_fuji(pixel_array: np.ndarray) -> np.ndarray:
    lin = 10 ** ((pixel_array / 4 - 2047) / 1024)
    rescaled = lin * 2**12 / lin.max()
    return rescaled.astype(np.uint16)


def _preprocess_fuji_mag(
    pixel_array: np.ndarray, pixel_spacing: float = None
) -> MammoMTFImage:
    # There is a border of saturated pixels around the mag image
    max_val = 2**14 - 1  # 14 bit images
    # Only take pixels that aren't saturated, with a small border
    image_indices = np.where(pixel_array != max_val)
    border_px = 20
    row_start = image_indices[0][0] + border_px
    row_end = image_indices[0][-1] - border_px
    col_start = image_indices[1][0] + border_px
    col_end = image_indices[1][-1] - border_px
    pixel_array = pixel_array[row_start:row_end, col_start:col_end]
    pixel_array = linearise_fuji(pixel_array)
    return MammoMTFImage(
        pixel_array,
        acquisition="mag",
        manufacturer="fuji",
        orientation="right",
        pixel_spacing=pixel_spacing,
    )


def preprocess_fuji(dcm: FileDataset, acquisition: str = None) -> np.ndarray:
    arr = dcm.pixel_array
    if not acquisition:
        img_type_header = dcm[0x0008, 0x0008].value
        if "TOMOSYNTHESIS" in img_type_header or "VOLUME" in img_type_header:
            pixel_spacing = get_pixel_spacing_tomo(dcm)
            mtf_image = autofocus_tomo(
                arr,
                manufacturer="fuji",
                orientation="right",
                pixel_spacing=pixel_spacing,
            )
        else:

            paddlevel = dcm[0x0018, 0x11A4].value
            pixel_spacing = get_pixel_spacing(dcm)
            if "MAG" in paddlevel:
                mtf_image = _preprocess_fuji_mag(
                    pixel_array=arr, pixel_spacing=pixel_spacing
                )
            else:
                arr = linearise_fuji(arr)
                mtf_image = MammoMTFImage(
                    arr,
                    acquisition="conventional",
                    manufacturer="fuji",
                    orientation="right",
                    pixel_spacing=pixel_spacing,
                )
    return mtf_image


def preprocess_ge(dcm: FileDataset, acquisition: str = None) -> MammoMTFImage:
    arr = dcm.pixel_array
    pixel_spacing = get_pixel_spacing(dcm)
    return MammoMTFImage(
        arr,
        manufacturer="ge",
        acquisition="conventional",
        orientation="right",
        pixel_spacing=pixel_spacing,
    )
