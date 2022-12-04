import re
import os
import tkinter as tk
from tkinter import ttk
import tkinterdnd2 as tkdnd2
from tkinterdnd2 import DND_FILES
import pydicom
from pywintypes import com_error
from dicom_preprocess import preprocess_dicom
from mtf import get_labelled_rois, calculate_mtf
from application_parameters import get_sample_spacing, excel_write_cell, get_edge_locations, get_excel_write_sheet, get_overwrite_cells
import numpy as np
from excel_write import excelkey2ind, write_values, get_active_app


class AppMTF(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.grid()
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self.files_dropped)
        ttk.Label(self, text='DICOM images to process:').grid(column=0, row=0, columnspan=2)
        self.dcm_queue = {}
        self.processing_book = None
        self.dcm_list = tk.Variable(value=[])
        # self.dcm_listbox = tk.Listbox(self, listvariable=self.dcm_list, width=30, height=10)
        self.dcm_listbox = tk.Listbox(self, listvariable=self.dcm_list, height=10, width=30)
        self.dcm_listbox.grid(column=0, row=1, columnspan=2)

        self.button_clear_all = tk.Button(self, text='Clear all', command=self.clear_queue)
        self.button_clear_all.grid(column=0, row=2)
        self.button_delete_selected = tk.Button(self, text='Delete selected', command=self.delete_selected)
        self.button_delete_selected.grid(column=1, row=2)

        self.calculate_button = tk.Button(self, text='Calculate MTF', command=self.process_queue)
        self.calculate_button.grid(column=0, row=3, columnspan=2)

        self.excel_label = ttk.Label(self, text='Open excel workbook:')
        self.excel_label.grid(column=0, row=4, columnspan=2)

        self.excel_app = get_active_app()
        self.workbook_options = ['No workbook selected.']
        if self.excel_app:
            [self.workbook_options.append(book.name) for book in self.excel_app.books]
            default_val = self.workbook_options[1]
        else:
            default_val = self.workbook_options[0]

        self.open_excel_var = tk.StringVar(self)
        self.open_excel_var.set(default_val)

        self.excel_option_menu = ttk.OptionMenu(self, self.open_excel_var, default_val, *self.workbook_options)
        self.excel_option_menu.grid(column=0, row=5, columnspan=2)
        self.update_workbook_options()

    def update_workbook_options(self):
        self.workbook_options = ['No workbook selected.']
        self.excel_app = get_active_app()
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
        print(filestr)
        re_fpaths = re.compile('\{[^}^{]*\}')
        fpaths1 = re.findall(re_fpaths, filestr)
        fpaths1 = [fpath.strip('{}') for fpath in fpaths1]
        print(fpaths1)
        fpaths2 = re.sub(re_fpaths, '', filestr)
        fpaths2 = fpaths2.split()
        print(fpaths2)
        fpaths = fpaths1 + fpaths2
        # fpaths = filestr.split()

        print(fpaths)
        # Only add a file to the queue if it is not already in queue.
        fpaths = (fpath for fpath in fpaths if fpath not in self.dcm_queue.values())
        for fpath in fpaths:
            _, fname = os.path.split(fpath)
            self.dcm_queue[fname] = fpath
            self.dcm_listbox.insert('end', fname)

    def process_queue(self):
        for _, fpath in self.dcm_queue.items():
            self.process_image(fpath)

    def delete_selected(self):
        selected = self.dcm_listbox.curselection()
        if selected:
            selected_idx = selected[0]
            selected_key = list(self.dcm_queue)[selected_idx]
            self.dcm_queue.pop(selected_key)
            self.dcm_listbox.delete(selected_idx)
            print(self.dcm_queue)

    def clear_queue(self):
        self.dcm_queue = {}
        self.dcm_listbox.delete(0, last=tk.END)
        print(self.dcm_queue)

    def process_image(self, dcmpath):
        dcm = pydicom.dcmread(dcmpath)
        dicom_data = preprocess_dicom(dcm)
        manufacturer = dicom_data['manufacturer']
        mode = dicom_data['mode']
        rois, rois_canny = get_labelled_rois(dicom_data['pixel_array'])

        sample_number = 105 # number of MTF samples to take
        edges = get_edge_locations(rois)
        for edge_position in edges:

            if edge_position in ('left', 'right'):
                edge_dir = 'vertical'
            elif edge_position in ('top', 'bottom'):
                edge_dir = 'horizontal'

            if edge_position in rois:
                edge_roi = rois[edge_position]
                edge_roi_canny = rois_canny[edge_position]
                sample_spacing = get_sample_spacing(manufacturer, mode)
                f, MTF = calculate_mtf(edge_roi, sample_spacing, edge_roi_canny,
                    edge_dir, sample_number)

                excel_cell = excel_write_cell(mode, edge_dir)
                write_array = np.array([f, MTF]).T

            else:
                excel_cell = excel_write_cell(mode, edge_dir)
                # Write empty array if intended edge to process not found.
                write_array = np.empty((sample_number, 2))
                write_array[:] = np.nan

            try:
                book_name = self.open_excel_var.get()
                write_sheet = get_excel_write_sheet()
                overwrite = get_overwrite_cells()
                sheet = self.excel_app.books[book_name].sheets[write_sheet]
                write_values(sheet, write_array, excel_cell, overwrite=overwrite)
            except com_error as e:
                print('Couldn\'t find the Resolution sheet')
        

if __name__ == '__main__':
    root = tkdnd2.Tk()
    root.title('MTF calculator')
    root.geometry('250x350')
    app = AppMTF(root)
    app.pack(expand=True)
    root.mainloop()


