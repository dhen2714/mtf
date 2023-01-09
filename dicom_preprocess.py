import cv2


def preprocess_dicom(dcm):
    """
    Preprocesses the DICOM header and image, returning a dicom data dictionary.
    data dictionary contains:
    'manufacturer' : manufacturer name
    'mode' : either tomo, mag, or conventional
    'tomo_slice' : in focus slice index if image is tomo recon
    'pixel_array' : 2D pixel image array
    """
    dicom_data = {}
    manufacturer = dcm[0x0008, 0x0070].value.lower()

    if 'hologic' in manufacturer:
        dicom_data['manufacturer'] = 'hologic'
        dicom_data = preprocess_hologic(dcm, dicom_data)
    elif 'ge' in manufacturer:
        dicom_data['manufacturer'] = 'ge'
        dicom_data = preprocess_ge(dcm, dicom_data)
    return dicom_data


def preprocess_hologic(dcm, data_dict):
    """
    Queries Hologic dicom header, returning either 'tomo', 'mag' or 
    'conventional'.
    """
    img_type_header = dcm[0x0008, 0x0008].value
    if 'TOMOSYNTHESIS' in img_type_header or 'VOLUME' in img_type_header:
        data_dict['mode'] = 'tomo'
        pixel_array = dcm.pixel_array
        tomo_slice = autofocus_tomo(pixel_array)
        data_dict['pixel_array'] = dcm.pixel_array[tomo_slice, ...]
        data_dict['tomo_slice'] = tomo_slice
    else:
        # Get value for (0018, 11a4) Paddle description
        paddleval = dcm[0x0018,0x11a4].value   
        pixel_array = dcm.pixel_array    
        if paddleval == '10CM MAG':
            data_dict['mode'] = 'mag'
            rowlims = (450, 2800)
        else:
            data_dict['mode'] = 'conventional'
            rowlims = (20, -20)
        data_dict['pixel_array'] = pixel_array[rowlims[0]:rowlims[1],:]
    return data_dict


def preprocess_ge(dcm, data_dict):
    data_dict['mode'] = 'conventional'
    data_dict['pixel_array'] = dcm.pixel_array
    return data_dict


def autofocus_tomo(tomo_recon):
    """
    Find the tomosynthesis slice in which the MTF edge is in focus.
    Uses variance of Laplacian as a metric for focus.
    Input: 3D tomosynthesis reconstruction pixel array.
    Output: Slice index number for in-focus slice.
    """
    max_lapvar_slice = 0
    max_lapvar = 0
    for i, tomo_slice in enumerate(tomo_recon):
        lapvar = cv2.Laplacian(tomo_slice, cv2.CV_64F).var()
        if lapvar > max_lapvar:
            max_lapvar_slice = i
            max_lapvar = lapvar
    return max_lapvar_slice