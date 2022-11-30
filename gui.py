import re
import os
import tkinter as tk
from tkinter import ttk
import tkinterdnd2 as tkdnd2
from tkinterdnd2 import DND_FILES
import xlwings as xw
import pydicom
from pywintypes import com_error
from dicom_preprocess import preprocess_dicom
from mtf import get_labelled_rois, calculate_mtf
from application_parameters import get_sample_spacing, excel_write_cell
import numpy as np


def excelkey2ind(excelkey):
    """
    Converts excel key in <letter, number> format to (row number, column number).
    E.g.
    excelkey2ind('E3') = (3, 5)
    """
    reg = re.search('\D+', excelkey)
    colxl, rowxl = excelkey[reg.start():reg.end()], excelkey[reg.end():]
    
    numletters = len(colxl)
    colnum = 0
    for i, letter in enumerate(colxl):
        power = numletters - i - 1
        colnum += (ord(letter.lower()) - 96)*(26**power)
        
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
        
    startrc = excelkey2ind(cell_key)
    endrow, endcol = (startrc[0] + array_shape[0] - 1), (startrc[1] + array_shape[1] - 1)
    values = np.array(xw_sheet.range(startrc, (endrow, endcol)).value)
    
    if values.any() and not overwrite:
        raise Exception('Values detected in cells.')
    else:
        xw_sheet.range(startrc, (endrow, endcol)).value = array
    
    return


class AppMTF(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid()
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.files_dropped)
        ttk.Label(self, text='DICOM images to process:').grid(column=0, row=0)
        self.dcm_queue = {}
        self.processing_book = None
        self.dcm_list = tk.Variable(value=[])
        self.dcm_listbox = tk.Listbox(self, listvariable=self.dcm_list, width=30, height=10)
        self.dcm_listbox.grid(column=0, row=1)

        self.calculate_button = tk.Button(self, text='Calculate MTF', command=self.process_queue)
        self.calculate_button.grid(column=0, row=2)

        self.excel_label = ttk.Label(self, text='Open excel workbook:')
        self.excel_label.grid(column=0, row=3)

        self.excel_app = xw.apps.active
        self.workbook_options = ['No workbook selected.']
        if self.excel_app:
            [self.workbook_options.append(book.name) for book in self.excel_app.books]
            default_val = self.workbook_options[1]
        else:
            default_val = self.workbook_options[0]

        self.open_excel_var = tk.StringVar(self)
        self.open_excel_var.set(default_val)

        self.excel_option_menu = ttk.OptionMenu(self, self.open_excel_var, default_val, *self.workbook_options)
        self.excel_option_menu.grid(column=0, row=4)
        self.update_workbook_options()

    def update_workbook_options(self):
        self.workbook_options = ['No workbook selected.']
        self.excel_app = xw.apps.active
        if self.excel_app:
            try:
                [self.workbook_options.append(book.name) for book in self.excel_app.books]
            except com_error as e:
                print('HANDLED')
                self.open_excel_var.set('No workbook selected.')
        else:
            self.open_excel_var.set('No workbook selected.')

        self.update_option_menu()
        self.after(2000, self.update_workbook_options)
        print(self.open_excel_var.get())

    def update_option_menu(self):
        """
        Update the options menu to display open workbooks.
        """
        menu = self.excel_option_menu['menu']
        menu.delete(0, 'end')
        for option in self.workbook_options:
            menu.add_command(label=option, 
                command=lambda value=option: self.open_excel_var.set(value))

    def files_dropped(self, event):
        filestr = event.data
        re_fpaths = re.compile('\{[^}^{]*\}')
        fpaths = re.findall(re_fpaths, filestr)
        fpaths = [fpath.strip('{}') for fpath in fpaths]

        for fpath in fpaths:
            _, fname = os.path.split(fpath)
            self.dcm_queue[fname] = fpath
            self.dcm_listbox.insert('end', fname)

    def process_queue(self):
        for _, fpath in self.dcm_queue.items():
            self.process_image(fpath)

    def process_image(self, dcmpath):
        dcm = pydicom.dcmread(dcmpath)
        dicom_data = preprocess_dicom(dcm)
        manufacturer = dicom_data['manufacturer']
        mode = dicom_data['mode']
        rois, rois_canny = get_labelled_rois(dicom_data['pixel_array'])

        sample_number = 105 # number of MTF samples to take
        edges = ['left', 'top']
        for edge_position in edges:
            if edge_position in ('left', 'right'):
                edge_dir = 'vertical'
            elif edge_position in ('top', 'bottom'):
                edge_dir = 'horizontal'
            edge_roi = rois[edge_position]
            edge_roi_canny = rois_canny[edge_position]
            sample_spacing = get_sample_spacing(manufacturer, mode)
            f, MTF = calculate_mtf(edge_roi, sample_spacing, edge_roi_canny,
                edge_dir, sample_number)

            excel_cell = excel_write_cell(mode, edge_dir)
            write_array = np.array([f, MTF]).T

            try:
                book_name = self.open_excel_var.get()
                sheet = xw.apps.active.books[book_name].sheets['Resolution']
                write_values(sheet, write_array, excel_cell)
            except com_error as e:
                print('Couldn\'t find the Resolution sheet')
        

if __name__ == '__main__':
    root = tkdnd2.Tk()
    app = AppMTF(root)
    app.mainloop()


