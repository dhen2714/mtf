import xlwings as xw
import numpy as np
import re


def get_active_app():
    return xw.apps.active


def get_active_sheet_name():
    return xw.sheets.active.name


def excelkey2ind(excelkey):
    """
    Converts excel key in <letter, number> format to (row number, column number).
    E.g.
    excelkey2ind('E3') = (3, 5)
    """
    reg = re.search("\D+", excelkey)
    colxl, rowxl = excelkey[reg.start() : reg.end()], excelkey[reg.end() :]

    numletters = len(colxl)
    colnum = 0
    for i, letter in enumerate(colxl):
        power = numletters - i - 1
        colnum += (ord(letter.lower()) - 96) * (26**power)

    rownum = int(rowxl)
    return rownum, colnum


def write_values(xw_sheet, array, cell_key, overwrite=False):
    """
    Write values from array to Excel sheet.
    The cell_key is an Excel column-row key.
    Assumes array is 2D. The array will be written to the sheet with cell_key
    being the first value in array, or the top-left corner value.
    """
    array_dims = array.ndim
    if array_dims == 2:
        array_shape = array.shape
    else:
        array_shape = (len(array), 1)

    if type(cell_key) is str:
        startrc = excelkey2ind(cell_key)
    else:
        startrc = cell_key

    endrow, endcol = (startrc[0] + array_shape[0] - 1), (
        startrc[1] + array_shape[1] - 1
    )
    values = np.array(xw_sheet.range(startrc, (endrow, endcol)).value)

    if values.any() and not overwrite:
        raise Exception("Values detected in cells.")
    else:
        xw_sheet.range(startrc, (endrow, endcol)).value = array

    return
