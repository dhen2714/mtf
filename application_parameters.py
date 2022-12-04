import os
import json
from excel_write import get_active_sheet_name


if os.path.exists('mtfcalc_params.json'):
    with open('mtfcalc_params.json', 'r') as f:
        parameter_dict = json.load(f)

    USE_DEFAULT = parameter_dict['use_default_parameters']
else:
    USE_DEFAULT = True

if USE_DEFAULT:
    HOLOGIC_SPACING = {
        'tomo' : 0.108,
        'mag' : 0.0349,
        'conventional' : 0.0635, 
    }
    GE_SPACING = {
        'conventional' : 0.1,
    }
    PROCESS_EDGES = 'default'
    WRITE_MODE = 'template'
    OVERWRITE_CELLS = False
    EXCEL_WRITE_SHEET = 'Resolution'
else:
    HOLOGIC_SPACING = parameter_dict['hologic_spacing']
    GE_SPACING = parameter_dict['ge_spacing']
    PROCESS_EDGES = parameter_dict['process_edge']
    WRITE_MODE = parameter_dict['write_mode']
    OVERWRITE_CELLS = parameter_dict['overwrite_cells']
    EXCEL_WRITE_SHEET = parameter_dict['excel_write_sheet']


def get_sample_spacing(manufacturer, mode):
    if manufacturer == 'hologic':
        spacing = HOLOGIC_SPACING[mode]
    elif manufacturer == 'ge':
        spacing = GE_SPACING[mode]
    return spacing


def excel_write_free():
    row_start = 3
    col_start = 1
    if PROCESS_EDGES == 'all':
        increment = 2
    else:
        increment = 2
    while True:
        yield (row_start, col_start)
        col_start += increment


EXCEL_WRITE_FREE = excel_write_free()


def excel_write_template(mode, edge_direction, mode_setup='top'):
    if mode == 'tomo':
        setup = f'tomo recon {mode_setup}'
    else:
        setup = mode

    if setup == 'conventional':
        if edge_direction == 'vertical':
            excel_key = 'AF8'
        elif edge_direction == 'horizontal':
            excel_key = 'AH8'
    elif setup == 'mag':
        if edge_direction == 'vertical':
            excel_key = 'AJ8'
        elif edge_direction == 'horizontal':
            excel_key = 'AL8'
    elif setup == 'tomo recon top':
        if edge_direction == 'vertical':
            excel_key = 'AV8'
        elif edge_direction == 'horizontal':
            excel_key = 'AX8'

    return excel_key


def excel_write_cell(mode, edge_direction, mode_setup='top'):
    if WRITE_MODE == 'template':
        excel_key = excel_write_template(mode, edge_direction, mode_setup)
    elif WRITE_MODE == 'free':
        excel_key = next(EXCEL_WRITE_FREE)
    return excel_key


def get_edge_locations(detected_edge_locations):
    """
    Return edge locations to calculate MTF for.
    """
    edge_locations = []
    if PROCESS_EDGES == 'default':
        if 'left' in detected_edge_locations:
            edge_locations.append('left')
        elif 'right' in detected_edge_locations:
            edge_locations.append('right')

        if 'top' in detected_edge_locations:
            edge_locations.append('top')
        elif 'bottom' in detected_edge_locations:
            edge_locations.append('bottom')
    elif PROCESS_EDGES in ['left', 'right', 'top', 'bottom']:
        edge_locations.append(PROCESS_EDGES)
    elif PROCESS_EDGES == 'all':
        edge_locations.extend(['left', 'right', 'top', 'bottom'])

    return edge_locations


def get_excel_write_sheet():
    if EXCEL_WRITE_SHEET == '_active':
        write_sheet = get_active_sheet_name()
    else:
        write_sheet = EXCEL_WRITE_SHEET
    return write_sheet


def get_overwrite_cells():
    return OVERWRITE_CELLS


def get_write_mode():
    return WRITE_MODE
    


