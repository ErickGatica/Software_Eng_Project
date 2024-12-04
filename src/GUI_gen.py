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
# Importing the functions from the Abs_gen.py script
from Abs_gen import spectrum, plot_spectra

# Importing funtion to generate Lookuptable
from h6py_lookuptABLE_NICO_gen import create_lookup_table
from Normie_fitting import configure_plots, load_configuration, initialize_hapi_db
from Normie_fitting import process_file, plot_results
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
        menubar = self.menuBar()
        viewMenu = menubar.addMenu('View')

        zoom_action = QAction('Zoom', self)
        zoom_action.triggered.connect(self.zoom_plot)
        viewMenu.addAction(zoom_action)

    def zoom_plot(self):
        self.toolbar.zoom()

    def init_absorption_tab(self):
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
        # Create the main layout for the fitting tab
        main_layout = QVBoxLayout()

        # Create a splitter for resizable input and plot areas
        splitter = QSplitter(Qt.Horizontal)

        # Group for inputs
        input_group_box = QGroupBox("Input Parameters")
        input_layout = QFormLayout()

        # Set a fixed width for the input boxes
        input_width = 200

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

        # Display the input group box
        input_group_box.setLayout(input_layout)
        self.fitting_button.clicked.connect(self.fiting_data)
        splitter.addWidget(input_group_box)

        # Adding window to displayed the parameters of the fitting that were obtained
        self.window_fitting = QLineEdit()
        self.window_fitting.setReadOnly(True)
        input_layout.addRow(QLabel("Fitting Progress"), self.window_fitting)

        # Group for fitting progress inputs
        fitting_progress_group_box = QGroupBox("Fitting Results")
        fitting_progress_layout = QFormLayout()

        '''

        # Filename
        self.filename_input_fp = QLineEdit()
        self.filename_input_fp.setFixedWidth(input_width)
        self.filename_label_fp = QLabel("Filename")
        fitting_progress_layout.addRow(self.filename_label_fp, self.filename_input_fp)

        # Results path
        self.results_path_input_fp = QLineEdit()
        self.results_path_input_fp.setFixedWidth(input_width)
        self.results_path_label_fp = QLabel("Results Path")
        self.results_path_button_fp = QPushButton("Browse")
        self.results_path_button_fp.clicked.connect(lambda: self.browse_folder(self.results_path_input_fp))
        results_path_layout_fp = QHBoxLayout()
        results_path_layout_fp.addWidget(self.results_path_input_fp)
        results_path_layout_fp.addWidget(self.results_path_button_fp)
        fitting_progress_layout.addRow(self.results_path_label_fp, results_path_layout_fp)

        # Results filename
        self.results_filename_input_fp = QLineEdit()
        self.results_filename_input_fp.setFixedWidth(input_width)
        self.results_filename_label_fp = QLabel("Results Filename")
        fitting_progress_layout.addRow(self.results_filename_label_fp, self.results_filename_input_fp)

        # Plot name
        self.plot_name_input_fp = QLineEdit()
        self.plot_name_input_fp.setFixedWidth(input_width)
        self.plot_name_label_fp = QLabel("Plot Name")
        fitting_progress_layout.addRow(self.plot_name_label_fp, self.plot_name_input_fp)

        # Linelist path
        self.linelist_path_input_fp = QLineEdit()
        self.linelist_path_input_fp.setFixedWidth(input_width)
        self.linelist_path_label_fp = QLabel("Linelist Path")
        self.linelist_path_button_fp = QPushButton("Browse")
        self.linelist_path_button_fp.clicked.connect(lambda: self.browse_folder(self.linelist_path_input_fp))
        linelist_path_layout_fp = QHBoxLayout()
        linelist_path_layout_fp.addWidget(self.linelist_path_input_fp)
        linelist_path_layout_fp.addWidget(self.linelist_path_button_fp)
        fitting_progress_layout.addRow(self.linelist_path_label_fp, linelist_path_layout_fp)

        '''

        # Data path
        self.data_path_input_fp = QLineEdit()
        self.data_path_input_fp.setFixedWidth(input_width)
        self.data_path_label_fp = QLabel("Data Path")
        self.data_path_button_fp = QPushButton("Browse")
        self.data_path_button_fp.clicked.connect(lambda: self.browse_file(self.data_path_input_fp))
        data_path_layout_fp = QHBoxLayout()
        data_path_layout_fp.addWidget(self.data_path_input_fp)
        data_path_layout_fp.addWidget(self.data_path_button_fp)
        fitting_progress_layout.addRow(self.data_path_label_fp, data_path_layout_fp)


        # Config file path of the file 
        self.config_file_input_fp = QLineEdit()
        self.config_file_input_fp.setFixedWidth(input_width)
        self.config_file_label_fp = QLabel("Config File Path")
        self.config_file_button_fp = QPushButton("Browse")
        self.config_file_button_fp.clicked.connect(lambda: self.browse_file(self.config_file_input_fp))
        config_file_layout_fp = QHBoxLayout()
        config_file_layout_fp.addWidget(self.config_file_input_fp)
        config_file_layout_fp.addWidget(self.config_file_button_fp)
        fitting_progress_layout.addRow(self.config_file_label_fp, config_file_layout_fp)

        # Fitting progress button
        self.fitting_progress_button = QPushButton("Fit Data HAPI")
        fitting_progress_layout.addRow(self.fitting_progress_button)

        # Display the fitting progress group box
        fitting_progress_group_box.setLayout(fitting_progress_layout)
        self.fitting_progress_button.clicked.connect(self.fit_data_hapi)
        splitter.addWidget(fitting_progress_group_box)

        # Adding window to displayed the parameters of the fitting that were obtained
        self.window_fitting_hapi = QLineEdit()
        self.window_fitting_hapi.setReadOnly(True)
        fitting_progress_layout.addRow(QLabel("Fitting Results"), self.window_fitting_hapi)


        # Creating canvas to plot fit and experimental data
        # Apply dark background style globally
        plt.style.use('dark_background')

        '''
        # Placeholder for the plot area
        self.fitting_plot_canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.fitting_plot_canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.fitting_plot_canvas)
        plot_frame = QFrame()
        plot_frame.setLayout(plot_layout)
        splitter.addWidget(plot_frame)
        '''
        # Add the splitter to the main layout
        main_layout.addWidget(splitter)
        

        # Set the main layout for the fitting tab
        self.fitting_tab.setLayout(main_layout)



    def init_lookuptable_tab(self):
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

        '''
        # Group for sliders
        sliders_group_box = QGroupBox("Plot Parameters")
        sliders_layout = QFormLayout()

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
        self.update_plot_button.clicked.connect(self.update_plot)
        sliders_layout.addRow(self.update_plot_button)

        # Set layout for the sliders group box
        sliders_group_box.setLayout(sliders_layout)
        '''

        # Layout to stack the input parameters and sliders group boxes vertically
        input_and_sliders_layout = QVBoxLayout()
        input_and_sliders_layout.addWidget(input_group_box)
        #input_and_sliders_layout.addWidget(sliders_group_box) # TO DO

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

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", f"Invalid input: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")
        
        # Returning data for further use
        return data

    def validate_input(self, input_field, field_name):
        value = input_field.text()
        print(f"Validating {field_name}: '{value}'")  # Debug statement
        if not value:
            raise ValueError(f"{field_name} cannot be empty.")
        try:
            return float(value)
        except ValueError:
            raise ValueError(f"{field_name} must be a valid number.")

    def clear_plot(self):
        ax = self.absorption_plot_canvas.figure.gca()
        ax.clear()
        self.absorption_plot_canvas.draw()
        print("Plot cleared.")
    
    def browse_folder(self, target_input):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            target_input.setText(folder_path)
    
    def browse_file(self, target_input):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")
        if file_path:
            target_input.setText(file_path)

    # Function to save the absorption data of the plot in a text file
    def save_data(self, data):
        nu = data.nu
        absorption = data.absorption
        label = str(data.molecule) + "_" + str(data.isotopologue) + "_" + str(data.T) + "_" + str(data.P) + "_" + str(data.molar) + "_" + str(data.length) + "_" + str(data.method) 
        with open("absorption_data.txt", "w") as f:
            for i in range(len(nu)):
                if i == 1:
                    f.write(f"labels\t{label}\n")
                else:
                    f.write(f"{nu[i]}\t{absorption[i]}\n")

    def update_progress(self, message):
        self.progress_window.setText(message)


    def generate_lookuptable(self):
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
            "wavenumber_array": np.linspace(float(self.min_wavenumber_input.text()), float(self.max_wavenumber_input.text()), 500),
            "output_csv_file": self.csv_name_input.text(),
        }
        
        self.update_progress("Generating.")
        # Generate the lookup table
        create_lookup_table(params)
        self.update_progress("Successfully.")

    def update_sliders(self):
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


    def update_plot(self):
        pass

    def fiting_data(self):
        pass    

    def fit_data_hapi(self):
        # Configure plot format
        configure_plots()
        # Load configuration
        config_path = self.config_file_input_fp.text()
        config_variables = load_configuration(config_path)
        # Initialize HAPI database if necessary
        initialize_hapi_db(config_variables)
        # Process files
        process_file(self.data_path_input_fp.text(), config_variables)
        # Once its done print Successfully in the window
        self.window_fitting_hapi.setText("Successfully.")
        

# Run the application
app = QApplication(sys.argv)
window = GUI()
window.show()
sys.exit(app.exec_())