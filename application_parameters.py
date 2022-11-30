HOLOGIC_SPACING = {
    'tomo' : 0.108,
    'mag' : 0.0349,
    'conventional' : 0.0635, 
}


GE_SPACING = {
    'conventional' : 0.1,
}


def get_sample_spacing(manufacturer, mode):
    if manufacturer == 'hologic':
        spacing = HOLOGIC_SPACING[mode]
    elif manufacturer == 'ge':
        spacing = GE_SPACING[mode]
    return spacing


def excel_write_cell(mode, edge_direction, mode_setup='top'):
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

