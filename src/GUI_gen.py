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
from PyQt5.QtWidgets import QMessageBox

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Importing the functions from the Abs_gen.py script
from Abs_gen import spectrum, plot_spectra


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

        # Generate button
        self.generate_button = QPushButton("Generate Absorption Spectra")
        self.generate_button.clicked.connect(self.generate_absorption_spectra)
        input_layout.addRow(self.generate_button)

        # Clear button
        self.clear_button = QPushButton("Clear Plot")
        self.clear_button.clicked.connect(self.clear_plot)
        input_layout.addRow(self.clear_button)

        input_group_box.setLayout(input_layout)
        splitter.addWidget(input_group_box)

        # Apply dark background style globally
        plt.style.use('dark_background')

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
        self.fitting_button = QPushButton("Fit Data")
        input_layout.addRow(self.fitting_button)

        # Display the input group box
        input_group_box.setLayout(input_layout)
        self.fitting_button.clicked.connect(self.fiting_data)
        splitter.addWidget(input_group_box)

        # Creating canvas to plot fit and experimental data
        # Apply dark background style globally
        plt.style.use('dark_background')

        # Placeholder for the plot area
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
        # Create the main layout for the lookuptable tab
        main_layout = QVBoxLayout()

        # Create a splitter for resizable input and plot areas
        splitter = QSplitter(Qt.Horizontal)

        # Group for inputs
        input_group_box = QGroupBox("Input Parameters")
        input_layout = QFormLayout()

        # Set a fixed width for the input boxes
        input_width = 200

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

        # Window to print the progress of the lookuptable generation
        self.progress_window = QLineEdit()
        self.progress_window.setReadOnly(True)
        input_layout.addRow(QLabel("Progress"), self.progress_window)

        # Button to start the lookup table generation
        self.generate_button = QPushButton("Generate Lookuptable")
        input_layout.addRow(self.generate_button)

        # Display the input group box
        input_group_box.setLayout(input_layout)
        splitter.addWidget(input_group_box)

        # Apply dark background style globally
        plt.style.use('dark_background')

        # Placeholder for the plot area
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

    def fiting_data(self):
        pass    

# Run the application
app = QApplication(sys.argv)
window = GUI()
window.show()
sys.exit(app.exec_())