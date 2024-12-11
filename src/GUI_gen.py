"""
This script generates the GUI for the generation of spectra data, lookuptables and fitting of the experimental data
There is a tab for each of the three functionalities
It uses the pyqt5 library to generate the GUI
"""
# Libraries
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout
from PyQt5.QtWidgets import QTabWidget, QLabel, QLineEdit, QPushButton
from PyQt5.QtWidgets import QComboBox, QFrame, QGroupBox, QMenuBar, QAction
from PyQt5.QtWidgets import QSplitter, QFormLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog, QPushButton
from PyQt5.QtWidgets import QMessageBox, QCheckBox, QSlider

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
# Importing the functions from the Abs_gen.py script
from Abs_gen import spectrum, plot_spectra

# Importing funtion to generate Lookuptable
from h6py_lookuptABLE_NICO_gen import create_lookup_table
from Normie_fitting import configure_plots, load_configuration, initialize_hapi_db
from Normie_fitting import process_file, plot_results
from Normal_fit_data import voigt_profile, multi_voigt 
# Importing the variables from the Variables.py script
# from Variables import args
# from Variables import args_dict

# Defining dictionary with the ID of the molecules
molecule_id_dict = {
    "H2O": 1,
    "CO2": 2,
    "CO": 5,
    "N2": 22,
    "O2": 7,
    "CH4": 6,
    "C2H6": 27,
    "H2": 45,
    "NO":8,
    "NO2":10
}


class GUI(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spectra Data Generator")
        self.setGeometry(100, 100, 800, 600)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.absorption_tab = QWidget()
        self.fitting_tab = QWidget()
        self.lookuptable_tab = QWidget()

        self.tabs.addTab(self.absorption_tab, "Absorption Spectra")
        self.tabs.addTab(self.fitting_tab, "Fitting Data")
        self.tabs.addTab(self.lookuptable_tab, "Lookuptable Generation")

        self.init_absorption_tab()
        self.init_fitting_tab()
        self.init_lookuptable_tab()

        self.create_menu()

    def create_menu(self):
        '''
        Create a menu bar with options to zoom the plot
        '''
        menubar = self.menuBar()
        viewMenu = menubar.addMenu('View')

        zoom_action = QAction('Zoom', self)
        zoom_action.triggered.connect(self.zoom_plot)
        viewMenu.addAction(zoom_action)

    def zoom_plot(self):
        '''
        Call the zoom function of the toolbar to enable zooming in the plot
        '''
        self.toolbar.zoom()

    def init_absorption_tab(self):
        '''
        Initialize the layout for the absorption spectra tab
        '''
        main_layout = QVBoxLayout()

        # Description and logo layout
        description_logo_layout = QHBoxLayout()

        description_label = QLabel("Ph.D. students: Gatica, Erick. Harris, Nico. Richter, Eric")
        font = QFont()
        font.setPointSize(14)
        description_label.setFont(font)
        logo_label = QLabel()
        pixmap = QPixmap("images/PLDL_logo.png")
        scaled_pixmap = pixmap.scaled(450, 450, Qt.KeepAspectRatio)
        logo_label.setPixmap(scaled_pixmap)

        description_layout = QVBoxLayout()
        description_layout.addWidget(description_label)

        description_logo_layout.addLayout(description_layout)
        description_logo_layout.addWidget(logo_label)

        main_layout.addLayout(description_logo_layout)

        # Input parameters and plot layout
        splitter = QSplitter(Qt.Horizontal)

        input_group_box = QGroupBox("Input Parameters")
        input_layout = QFormLayout()

        # Set a fixed width for the input boxes
        input_width = 200

        # Molecule
        self.molecule_input = QComboBox()
        self.molecule_input.addItems(["H2O", "CO2", "CO", "N2", "O2", "CH4", "H2", "NO", "NO2"])
        self.molecule_input.setFixedWidth(input_width)
        self.molecule_label = QLabel("Molecule:")
        input_layout.addRow(self.molecule_label, self.molecule_input)

        # Isotopologue
        self.isotopologue_input = QComboBox()
        self.isotopologue_input.addItems(["1", "2", "3"])
        self.isotopologue_input.setFixedWidth(input_width)
        self.isotopologue_label = QLabel("Isotopologue")
        input_layout.addRow(self.isotopologue_label, self.isotopologue_input)

        # Temperature
        self.temp_input = QLineEdit()
        self.temp_input.setFixedWidth(input_width)
        self.temp_label = QLabel("Temperature (K)")
        input_layout.addRow(self.temp_label, self.temp_input)

        # Pressure
        self.pressure_input = QLineEdit()
        self.pressure_input.setFixedWidth(input_width)
        self.pressure_label = QLabel("Pressure (atm)")
        input_layout.addRow(self.pressure_label, self.pressure_input)

        # Molar Fraction
        self.molar_fraction_input = QLineEdit()
        self.molar_fraction_input.setFixedWidth(input_width)
        self.molar_fraction_label = QLabel("Molar Fraction")
        input_layout.addRow(self.molar_fraction_label, self.molar_fraction_input)

        # Path Length
        self.length_input = QLineEdit()
        self.length_input.setFixedWidth(input_width)
        self.length_label = QLabel("Path Length (cm)")
        input_layout.addRow(self.length_label, self.length_input)

        # Minimum Wavenumber
        self.wavenumber_min_input = QLineEdit()
        self.wavenumber_min_input.setFixedWidth(input_width)
        self.wavenumber_min_label = QLabel("Minimum Wavenumber (cm-1)")
        input_layout.addRow(self.wavenumber_min_label, self.wavenumber_min_input)

        # Maximum Wavenumber
        self.wavenumber_max_input = QLineEdit()
        self.wavenumber_max_input.setFixedWidth(input_width)
        self.wavenumber_max_label = QLabel("Maximum Wavenumber (cm-1)")
        input_layout.addRow(self.wavenumber_max_label, self.wavenumber_max_input)

        # Wavenumber Step
        self.wavenumber_step_input = QLineEdit()
        self.wavenumber_step_input.setFixedWidth(input_width)
        self.wavenumber_step_label = QLabel("Wavenumber Step (cm-1)")
        input_layout.addRow(self.wavenumber_step_label, self.wavenumber_step_input)

        # Method
        self.method_input = QComboBox()
        self.method_input.addItems(["HT", "V", "L", "D"])
        self.method_input.setFixedWidth(input_width)
        self.method_label = QLabel("Method")
        input_layout.addRow(self.method_label, self.method_input)

        # Apply dark background style globally
        plt.style.use('dark_background')

        # Checkbox to enable or disable saving data of absorption in text file
        self.save_data_checkbox = QCheckBox("Save Generated Data in text file")
        input_layout.addRow(self.save_data_checkbox)


        # Generate button
        self.generate_button = QPushButton("Generate Absorption Spectra")
        self.generate_button.clicked.connect(self.generate_absorption_spectra)
        input_layout.addRow(self.generate_button)

        # Clear button
        self.clear_button = QPushButton("Clear Plot")
        self.clear_button.clicked.connect(self.clear_plot)
        input_layout.addRow(self.clear_button)

        # Save button
        '''
        self.save_button = QPushButton("Save Data")
        self.save_button.clicked.connect(self.save_data)
        input_layout.addRow(self.save_button)
        '''
        
        input_group_box.setLayout(input_layout)
        splitter.addWidget(input_group_box)


        # Placeholder for the plot area
        self.absorption_plot_canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.absorption_plot_canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.absorption_plot_canvas)
        plot_frame = QFrame()
        plot_frame.setLayout(plot_layout)
        splitter.addWidget(plot_frame)

        main_layout.addWidget(splitter)
        self.absorption_tab.setLayout(main_layout)

    def init_fitting_tab(self):
        '''
        Initialize the layout for the fitting tab
        '''
        # Create the main layout for the fitting tab
        main_layout = QHBoxLayout()  # Horizontal layout to place inputs and plot side by side

        # Create a splitter for resizable input and plot areas
        splitter = QSplitter(Qt.Horizontal)

        # Set a fixed width for the input boxes
        input_width = 200

        # TODO : Integration lookup table fitting
        '''
        # Group for input parameters
        input_group_box = QGroupBox("Lookuptable Fitting")
        input_layout = QFormLayout()

        # Data path
        self.data_path_input = QLineEdit()
        self.data_path_input.setFixedWidth(input_width)
        self.data_path_label = QLabel("Data Path")
        self.data_path_button = QPushButton("Browse")
        self.data_path_button.clicked.connect(lambda: self.browse_folder(self.data_path_input))
        data_path_layout = QHBoxLayout()
        data_path_layout.addWidget(self.data_path_input)
        data_path_layout.addWidget(self.data_path_button)
        input_layout.addRow(self.data_path_label, data_path_layout)

        # Filename
        self.filename_input = QLineEdit()
        self.filename_input.setFixedWidth(input_width)
        self.filename_label = QLabel("Filename")
        input_layout.addRow(self.filename_label, self.filename_input)

        # Results path
        self.results_path_input = QLineEdit()
        self.results_path_input.setFixedWidth(input_width)
        self.results_path_label = QLabel("Results Path")
        self.results_path_button = QPushButton("Browse")
        self.results_path_button.clicked.connect(lambda: self.browse_folder(self.results_path_input))
        results_path_layout = QHBoxLayout()
        results_path_layout.addWidget(self.results_path_input)
        results_path_layout.addWidget(self.results_path_button)
        input_layout.addRow(self.results_path_label, results_path_layout)

        # Results filename
        self.results_filename_input = QLineEdit()
        self.results_filename_input.setFixedWidth(input_width)
        self.results_filename_label = QLabel("Results Filename")
        input_layout.addRow(self.results_filename_label, self.results_filename_input)

        # Plot name
        self.plot_name_input = QLineEdit()
        self.plot_name_input.setFixedWidth(input_width)
        self.plot_name_label = QLabel("Plot Name")
        input_layout.addRow(self.plot_name_label, self.plot_name_input)

        # Linelist path
        self.linelist_path_input = QLineEdit()
        self.linelist_path_input.setFixedWidth(input_width)
        self.linelist_path_label = QLabel("Linelist Path")
        self.linelist_path_button = QPushButton("Browse")
        self.linelist_path_button.clicked.connect(lambda: self.browse_folder(self.linelist_path_input))
        linelist_path_layout = QHBoxLayout()
        linelist_path_layout.addWidget(self.linelist_path_input)
        linelist_path_layout.addWidget(self.linelist_path_button)
        input_layout.addRow(self.linelist_path_label, linelist_path_layout)

        # Fitting button
        self.fitting_button = QPushButton("Fit Data Lookuptable")
        input_layout.addRow(self.fitting_button)
        self.fitting_button.clicked.connect(self.fiting_data)

        # Window to display fitting progress
        self.window_fitting = QLineEdit()
        self.window_fitting.setReadOnly(True)
        input_layout.addRow(QLabel("Fitting Progress"), self.window_fitting)

        # Set layout for input group box
        input_group_box.setLayout(input_layout)
        '''
        # Group for fitting results
        fitting_progress_group_box = QGroupBox("Normal Fitting")
        fitting_progress_layout = QFormLayout()

        # Data path for experimental data
        self.data_path_input_fp = QLineEdit()
        self.data_path_input_fp.setFixedWidth(input_width)
        self.data_path_label_fp = QLabel("Data Path")
        self.data_path_button_fp = QPushButton("Browse")
        self.data_path_button_fp.clicked.connect(lambda: self.browse_file(self.data_path_input_fp))
        data_path_layout_fp = QHBoxLayout()
        data_path_layout_fp.addWidget(self.data_path_input_fp)
        data_path_layout_fp.addWidget(self.data_path_button_fp)
        fitting_progress_layout.addRow(self.data_path_label_fp, data_path_layout_fp)

        # Inputs for initial guess
        self.initial_guess_input = QLineEdit()
        self.initial_guess_input.setFixedWidth(input_width)
        self.initial_guess_label = QLabel("Initial Guess, N peak: ampi, centeri, sigmai, gammai, i = 1, N")
        fitting_progress_layout.addRow(self.initial_guess_label, self.initial_guess_input)

        # Fitting progress button
        self.fitting_progress_button = QPushButton("Fit Data")
        fitting_progress_layout.addRow(self.fitting_progress_button)
        self.fitting_progress_button.clicked.connect(self.fit_data_hapi)

        # Window to display fitting results
        self.window_fitting_hapi = QLineEdit()
        self.window_fitting_hapi.setReadOnly(True)
        fitting_progress_layout.addRow(QLabel("Fitting Results"), self.window_fitting_hapi)

        # Clear button
        self.clear_button_fp = QPushButton("Clear Plot")  
        fitting_progress_layout.addRow(self.clear_button_fp)
        self.clear_button_fp.clicked.connect(self.clear_plot_fitting)

        # Set layout for fitting progress group box
        fitting_progress_group_box.setLayout(fitting_progress_layout)

        # Create a vertical layout to stack both group boxes
        input_fitting_layout = QVBoxLayout()
        #input_fitting_layout.addWidget(input_group_box) # Reactivate with lookup
        input_fitting_layout.addWidget(fitting_progress_group_box)

        # Create a container for the left panel and add it to the splitter
        input_fitting_frame = QFrame()
        input_fitting_frame.setLayout(input_fitting_layout)
        splitter.addWidget(input_fitting_frame)

        # Plot area
        plt.style.use('dark_background')  # Apply dark background style globally
        self.fitting_plot_canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.fitting_plot_canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.fitting_plot_canvas)
        plot_frame = QFrame()
        plot_frame.setLayout(plot_layout)
        splitter.addWidget(plot_frame)

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Set the main layout for the fitting tab
        self.fitting_tab.setLayout(main_layout)




    def init_lookuptable_tab(self):
        '''
        Initialize the layout for the lookuptable generation tab
        '''
        # Create the main layout for the lookuptable tab
        main_layout = QVBoxLayout()

        # Create a splitter for resizable input and plot areas
        splitter = QSplitter(Qt.Horizontal)

        # Group for inputs
        input_group_box = QGroupBox("Input Parameters")
        input_layout = QFormLayout()

        # Set a fixed width for the input boxes
        input_width = 200

        # Molecule for Lookuptable generation
        self.molecule1_lookup_input = QComboBox()
        self.molecule1_lookup_input.addItems(["H2O", "CO2", "CO", "N2", "O2", "CH4", "H2", "NO", "NO2"])
        self.molecule1_lookup_input.setFixedWidth(input_width)
        self.molecule1_lookup_label = QLabel("Molecule:")
        input_layout.addRow(self.molecule1_lookup_label, self.molecule1_lookup_input)

        # Min temperature for Lookuptable
        self.min_temp_input = QLineEdit()
        self.min_temp_input.setFixedWidth(input_width)
        self.min_temp_label = QLabel("Min Temperature (K)")
        input_layout.addRow(self.min_temp_label, self.min_temp_input)

        # Max temperature for Lookuptable
        self.max_temp_input = QLineEdit()
        self.max_temp_input.setFixedWidth(input_width)
        self.max_temp_label = QLabel("Max Temperature (K)")
        input_layout.addRow(self.max_temp_label, self.max_temp_input)

        # Resolution for temperature
        self.resolution_temp_input = QLineEdit()
        self.resolution_temp_input.setFixedWidth(input_width)
        self.resolution_temp_label = QLabel("Resolution Temperature (K)")
        input_layout.addRow(self.resolution_temp_label, self.resolution_temp_input)

        # Min mole fraction for Lookuptable
        self.min_mole_fraction_input = QLineEdit()
        self.min_mole_fraction_input.setFixedWidth(input_width)
        self.min_mole_fraction_label = QLabel("Min Mole Fraction")
        input_layout.addRow(self.min_mole_fraction_label, self.min_mole_fraction_input)

        # Max mole fraction for Lookuptable
        self.max_mole_fraction_input = QLineEdit()
        self.max_mole_fraction_input.setFixedWidth(input_width)
        self.max_mole_fraction_label = QLabel("Max Mole Fraction")
        input_layout.addRow(self.max_mole_fraction_label, self.max_mole_fraction_input)

        # Resolution for mole fraction
        self.resolution_mole_fraction_input = QLineEdit()
        self.resolution_mole_fraction_input.setFixedWidth(input_width)
        self.resolution_mole_fraction_label = QLabel("Resolution Mole Fraction")
        input_layout.addRow(self.resolution_mole_fraction_label, self.resolution_mole_fraction_input)

        # Pressure for Lookuptable
        self.lookup_pressure_input = QLineEdit()
        self.lookup_pressure_input.setFixedWidth(input_width)
        self.lookup_pressure_label = QLabel("Pressure (atm)")
        input_layout.addRow(self.lookup_pressure_label, self.lookup_pressure_input)
        
        # Path length for Lookuptable
        self.lookup_length_input = QLineEdit()
        self.lookup_length_input.setFixedWidth(input_width)
        self.lookup_length_label = QLabel("Path Length (cm)")
        input_layout.addRow(self.lookup_length_label, self.lookup_length_input)


        # Shift range +- this value
        self.shift_range_input = QLineEdit()
        self.shift_range_input.setFixedWidth(input_width)
        self.shift_range_label = QLabel("Shift Range +-")
        input_layout.addRow(self.shift_range_label, self.shift_range_input)

        # Min wavenumber for Lookuptable
        self.min_wavenumber_input = QLineEdit()
        self.min_wavenumber_input.setFixedWidth(input_width)
        self.min_wavenumber_label = QLabel("Min Wavenumber (cm-1)")
        input_layout.addRow(self.min_wavenumber_label, self.min_wavenumber_input)

        # Max wavenumber for Lookuptable
        self.max_wavenumber_input = QLineEdit()
        self.max_wavenumber_input.setFixedWidth(input_width)
        self.max_wavenumber_label = QLabel("Max Wavenumber (cm-1)")
        input_layout.addRow(self.max_wavenumber_label, self.max_wavenumber_input)

        # Resolution for wavenumber
        self.resolution_wavenumber_input = QLineEdit()
        self.resolution_wavenumber_input.setFixedWidth(input_width)
        self.resolution_wavenumber_label = QLabel("Resolution Wavenumber (cm-1)")
        input_layout.addRow(self.resolution_wavenumber_label, self.resolution_wavenumber_input)

        # Input of the name for the csv file 
        self.csv_name_input = QLineEdit()
        self.csv_name_input.setFixedWidth(input_width)
        self.csv_name_label = QLabel("Name of the csv file")
        input_layout.addRow(self.csv_name_label, self.csv_name_input)


        # Window to print the progress of the lookuptable generation
        self.progress_window = QLineEdit()
        self.progress_window.setReadOnly(True)
        input_layout.addRow(QLabel("Progress"), self.progress_window)

        # Button to start the lookup table generation
        self.generate1_button = QPushButton("Generate Lookuptable")
        self.generate1_button.clicked.connect(self.generate_lookuptable)
        input_layout.addRow(self.generate1_button)
    
        # Display the input group box
        input_group_box.setLayout(input_layout)

        
        # Group for sliders
        sliders_group_box = QGroupBox("Plot Parameters")
        sliders_layout = QFormLayout()

        # Getting path of lookuptable file with browser function and button
        self.lookuptable_path_input = QLineEdit()
        self.lookuptable_path_input.setFixedWidth(input_width)
        self.lookuptable_path_label = QLabel("Lookuptable Path")
        self.lookuptable_path_button = QPushButton("Browse")
        self.lookuptable_path_button.clicked.connect(lambda: self.browse_file(self.lookuptable_path_input))
        lookuptable_path_layout = QHBoxLayout()
        lookuptable_path_layout.addWidget(self.lookuptable_path_input)
        lookuptable_path_layout.addWidget(self.lookuptable_path_button)
        sliders_layout.addRow(self.lookuptable_path_label, lookuptable_path_layout)

        # Temperature slider with default range
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setMinimum(300)  # Default minimum temperature
        self.temp_slider.setMaximum(1000)  # Default maximum temperature
        self.temp_slider.setValue(650)
        self.temp_slider.setTickPosition(QSlider.TicksBelow)
        self.temp_slider.setTickInterval(100)
        sliders_layout.addRow(QLabel("Temperature (K):"), self.temp_slider)

        # Mole Fraction slider with default range
        self.mole_fraction_slider = QSlider(Qt.Horizontal)
        self.mole_fraction_slider.setMinimum(0)  # Default minimum mole fraction
        self.mole_fraction_slider.setMaximum(100)  # Represented as percentage
        self.mole_fraction_slider.setValue(50)
        self.mole_fraction_slider.setTickPosition(QSlider.TicksBelow)
        self.mole_fraction_slider.setTickInterval(10)
        sliders_layout.addRow(QLabel("Molar Fraction"), self.mole_fraction_slider)

        # Connect input fields to dynamically update sliders
        self.min_temp_input.textChanged.connect(self.update_sliders)
        self.max_temp_input.textChanged.connect(self.update_sliders)
        self.resolution_temp_input.textChanged.connect(self.update_sliders)
        self.min_mole_fraction_input.textChanged.connect(self.update_sliders)
        self.max_mole_fraction_input.textChanged.connect(self.update_sliders)
        self.resolution_mole_fraction_input.textChanged.connect(self.update_sliders)


        # Button to update the plot
        self.update_plot_button = QPushButton("Update Plot")
        self.update_plot_button.clicked.connect(self.plot_lookuptable)
        sliders_layout.addRow(self.update_plot_button)

        # Button to clear the plot
        self.clear_plot_button = QPushButton("Clear Plot")
        self.clear_plot_button.clicked.connect(self.clear_plot_lookuptable)
        sliders_layout.addRow(self.clear_plot_button)


        # Set layout for the sliders group box
        sliders_group_box.setLayout(sliders_layout)
        

        # Layout to stack the input parameters and sliders group boxes vertically
        input_and_sliders_layout = QVBoxLayout()
        input_and_sliders_layout.addWidget(input_group_box)
        input_and_sliders_layout.addWidget(sliders_group_box) # TO DO

        # Container widget for input and sliders
        input_container = QFrame()
        input_container.setLayout(input_and_sliders_layout)
        splitter.addWidget(input_container)

        # Matplotlib plot area
        self.lookup_plot_canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.lookup_plot_canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.lookup_plot_canvas)
        plot_frame = QFrame()
        plot_frame.setLayout(plot_layout)
        splitter.addWidget(plot_frame)

        # Add the splitter to the main layout
        main_layout.addWidget(splitter)

        # Set the main layout for the lookuptable tab
        self.lookuptable_tab.setLayout(main_layout)

        # Connect sliders to the plotting function
        #self.temp_slider.valueChanged.connect(self.update_plot)
        #self.mole_fraction_slider.valueChanged.connect(self.update_plot)

    def generate_absorption_spectra(self):
        '''
        Generate the absorption spectra based on the input parameters
        '''
        try:
            molecule = self.molecule_input.currentText()
            # Molecule Id and isotopologue have to be a number
            molecule_id = molecule_id_dict[molecule]
            isotopologue = int(self.isotopologue_input.currentText())

            # Validate and convert input values
            T = self.validate_input(self.temp_input, "Temperature")
            P = self.validate_input(self.pressure_input, "Pressure")
            molar = self.validate_input(self.molar_fraction_input, "Molar Fraction")
            length = self.validate_input(self.length_input, "Path Length")
            wavenumber_min = self.validate_input(self.wavenumber_min_input, "Minimum Wavenumber")
            wavenumber_max = self.validate_input(self.wavenumber_max_input, "Maximum Wavenumber")
            wavestep = self.validate_input(self.wavenumber_step_input, "Wavenumber Step")
            method_ = self.method_input.currentText()

            print(f"Temperature: {T}, Pressure: {P}, Molar Fraction: {molar}, Path Length: {length}")
            print(f"Minimum Wavenumber: {wavenumber_min}, Maximum Wavenumber: {wavenumber_max}, Wavenumber Step: {wavestep}")

            # Generating the absorption spectra
            data = spectrum(P, T, length, wavenumber_min, wavenumber_max, molecule_id, isotopologue, method_, wavestep, molar)
            # Plotting the absorption spectra in the canvas of the window
            ax = self.absorption_plot_canvas.figure.gca()
            plot_spectra(ax, data)
            self.absorption_plot_canvas.figure.tight_layout()  # Adjust the layout
            self.absorption_plot_canvas.draw()
            print("Generating absorption spectra...")

            # Saving the data in text file if the checkbox is checked
            if self.save_data_checkbox.isChecked():
                self.save_data(data)
        
        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        
        # Returning data for further use
        return data

    def validate_input(self, input_field, field_name):
        '''
        Validate the input field and convert it to a float
        '''
        value = input_field.text()
        print(f"Validating {field_name}: '{value}'")  # Debug statement
        if not value:
            raise ValueError(f"{field_name} cannot be empty.")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{field_name} must be a valid number.")

    def clear_plot(self):
        '''
        Clear the plot area
        '''
        ax = self.absorption_plot_canvas.figure.gca()
        ax.clear()
        self.absorption_plot_canvas.draw()
        print("Plot cleared.")
    
    def clear_plot_fitting(self):
        '''
        Clear the plot area
        '''
        ax = self.fitting_plot_canvas.figure.gca()
        ax.clear()
        self.fitting_plot_canvas.draw()
        print("Plot cleared.")
    
    def clear_plot_lookuptable(self): 
        '''
        Clear the plot area
        '''
        ax = self.lookup_plot_canvas.figure.gca()
        ax.clear()
        self.lookup_plot_canvas.draw()
        print("Plot cleared.")

    def browse_folder(self, target_input):
        '''
        Browse for a folder and set the path in the target input field
        
        Inputs:
            target_input (QLineEdit): Input field to set the folder path
        '''
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            target_input.setText(folder_path)
    
    def browse_file(self, target_input):
        '''
        Browse for a file and set the path in the target input field
        
        Inputs:
            target_input (QLineEdit): Input field to set the file path  
        '''
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            target_input.setText(file_path)

    # Function to save the absorption data of the plot in a text file
    def save_data(self, data):
        '''
        Save the absorption data of the plot in a text file
        every time the checkbox is checked
        '''
        nu = data.nu
        absorption = data.absorption
        # Label using data class and input of molar fraction in GUI
        label = str(data.molecule) + "_" + str(data.temper) + "K_" + str(data.pressure) + "atm_" + str(self.molar_fraction_input.text()) + "MF"
        # Creating the text file with the absorption data and saving it in the folder
        with open(f"absorption_data_{label}.txt", "w") as file:
            file.write("Wavenumber (cm-1), Absorption\n")
            for i in range(len(nu)):
                file.write(f"{nu[i]}, {absorption[i]}\n")
        

    def update_progress(self, message):
        '''
        Update the progress window with a message
        '''
        self.progress_window.setText(message)


    def generate_lookuptable(self):
        '''
        Function to generate the lookuptable for the absorption spectra of the molecules
        '''
        # Creating dictionary with the parameters that create_lookup_table needs
        params ={
            "molecule_name": self.molecule1_lookup_input.currentText(),
            "hitran_id": molecule_id_dict[self.molecule1_lookup_input.currentText()],   
            "isotopologue_id": 1,
            "wavenumber_start": float(self.min_wavenumber_input.text()),
            "wavenumber_end": float(self.max_wavenumber_input.text()),
            "mole_fraction_start": float(self.min_mole_fraction_input.text()),
            "mole_fraction_end": float(self.max_mole_fraction_input.text()),
            "mole_fraction_resolution": float(self.resolution_mole_fraction_input.text()),
            "mole_fractions": np.arange(float(self.min_mole_fraction_input.text()), float(self.max_mole_fraction_input.text()), float(self.resolution_mole_fraction_input.text())),
            "min_temperature": float(self.min_temp_input.text()),
            "max_temperature": float(self.max_temp_input.text()),
            "temperature_resolution": float(self.resolution_temp_input.text()),
            "temperatures": np.arange(float(self.min_temp_input.text()), float(self.max_temp_input.text()), float(self.resolution_temp_input.text())),
            "pressure": float(self.lookup_pressure_input.text()),  
            "pathlength": float(self.lookup_length_input.text()),
            "spectral_shift": float(self.shift_range_input.text()),
            "fit_start": float(self.min_wavenumber_input.text()),
            "fit_end": float(self.max_wavenumber_input.text()),
            "wavenumber_array": np.linspace(float(self.min_wavenumber_input.text()), float(self.max_wavenumber_input.text()), float(self.resolution_wavenumber_input.text())),
            "output_csv_file": self.csv_name_input.text(),
        }
        
        self.update_progress("Generating.")
        # Generate the lookup table
        create_lookup_table(params)
        self.update_progress("Successfully.")

    def update_sliders(self):
        '''
        Update the sliders based on the input values
        '''
        try:
            # Update temperature slider
            min_temp = int(self.min_temp_input.text()) if self.min_temp_input.text() else 300
            max_temp = int(self.max_temp_input.text()) if self.max_temp_input.text() else 3000
            resolution_temp = int(self.resolution_temp_input.text()) if self.resolution_temp_input.text() else 100
            self.temp_slider.setMinimum(min_temp)
            self.temp_slider.setMaximum(max_temp)
            self.temp_slider.setValue(min_temp)
            self.temp_slider.setTickInterval(resolution_temp)

            # Update mole fraction slider
            min_mole_fraction = int(self.min_mole_fraction_input.text()) if self.min_mole_fraction_input.text() else 0
            max_mole_fraction = int(self.max_mole_fraction_input.text()) if self.max_mole_fraction_input.text() else 100
            resolution_mole_fraction = int(self.resolution_mole_fraction_input.text()) if self.resolution_mole_fraction_input.text() else 10
            self.mole_fraction_slider.setMinimum(min_mole_fraction)
            self.mole_fraction_slider.setMaximum(max_mole_fraction)
            self.mole_fraction_slider.setValue(min_mole_fraction)
            self.mole_fraction_slider.setTickInterval(resolution_mole_fraction)

        except ValueError:
            print("Invalid input for slider range. Please enter numeric values.")
    
    def plot_lookuptable(self):
        """
        Plot the lookup table based on the selected temperature and mole fraction
        """
        # Read the data from the input file
        data = pd.read_csv(self.lookuptable_path_input.text())

        # Extract relevant columns
        wavenumber = data['Wavenumber (cm⁻¹)'].values
        absorption = data['Absorption Coefficient'].values
        mole_fraction = data['Mole Fraction'].values
        temperature = data['Temperature (K)'].values

        # Update slider range inputs based on unique values in the data
        unique_temps = sorted(set(temperature))
        unique_mole_fractions = sorted(set(mole_fraction))

        self.min_temp_input.setText(str(min(unique_temps)))
        self.max_temp_input.setText(str(max(unique_temps)))
        self.resolution_temp_input.setText(str(len(unique_temps)))

        self.min_mole_fraction_input.setText(str(min(unique_mole_fractions)))
        self.max_mole_fraction_input.setText(str(max(unique_mole_fractions)))
        self.resolution_mole_fraction_input.setText(str(len(unique_mole_fractions)))

        # Update sliders with appropriate ranges
        self.temp_slider.setMinimum(0)
        self.temp_slider.setMaximum(len(unique_temps) - 1)
        self.temp_slider.setSingleStep(1)

        self.mole_fraction_slider.setMinimum(0)
        self.mole_fraction_slider.setMaximum(len(unique_mole_fractions) - 1)
        self.mole_fraction_slider.setSingleStep(1)

        # Get current slider positions and corresponding temperature and mole fraction
        temp_index = self.temp_slider.value()
        mole_fraction_index = self.mole_fraction_slider.value()
        selected_temp = unique_temps[temp_index]
        selected_mole_fraction = unique_mole_fractions[mole_fraction_index]

        # Filter data for the selected temperature and mole fraction
        mask = (temperature == selected_temp) & (mole_fraction == selected_mole_fraction)

        # Clear previous plots
        ax = self.lookup_plot_canvas.figure.gca()
        ax.clear()

        # Plot the filtered data
        if mask.any():
            ax.plot(
                wavenumber[mask],
                absorption[mask],
                label=f"T: {selected_temp} K, Mole Fraction: {selected_mole_fraction}"
            )
            ax.set_xlabel("Wavenumber (cm⁻¹)")
            ax.set_ylabel("Absorption Coefficient")
            ax.legend()
        else:
            ax.text(
                0.5, 0.5,
                "No data available for the selected parameters",
                transform=ax.transAxes,
                ha='center', va='center'
            )

        # Apply tight layout and redraw the canvas
        self.lookup_plot_canvas.figure.tight_layout()
        self.lookup_plot_canvas.draw()

    def update_plot(self):
        pass

    def fiting_data(self):
        pass    

    def fit_data_hapi(self):
        # Creating the array with the initial guess to use curve_fit function
        initial_guess = [float(i) for i in self.initial_guess_input.text().split(",")]
        # Getting the data from the file
        data_path = self.data_path_input_fp.text()
        data = pd.read_csv(data_path)
        frequencies = data['Frequency'].values
        intensities = data['Intensity'].values
        # Fit the data
        popt, pcov = curve_fit(multi_voigt, frequencies, intensities, p0=initial_guess)
        
        # Plot the results in the canvas of the tab
        ax = self.fitting_plot_canvas.figure.gca()
        ax.plot(frequencies, intensities, label="Experimental Data")
        ax.plot(frequencies, multi_voigt(frequencies, *popt), label="Fitted Data", linewidth=2)
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Absorption Coefficient")
        ax.legend()
        self.fitting_plot_canvas.figure.tight_layout()
        self.fitting_plot_canvas.draw()


        
        # Once its done print Successfully in the window
        self.window_fitting_hapi.setText("Successfully.")
        

# Run the application
app = QApplication(sys.argv)
window = GUI()
window.show()
sys.exit(app.exec_())