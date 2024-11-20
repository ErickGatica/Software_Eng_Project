"""
This script generates the GUI for the generation of spectra data, lookuptables and fitting of the experimental data
There is a tab for each of the three functionalities
It uses the pyqt5 library to generate the GUI
"""
# Libraries
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QLineEdit, QPushButton, QComboBox, QFrame, QGroupBox, QMenuBar, QAction, QSplitter
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
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
        input_layout = QVBoxLayout()

        molecule_layout = QHBoxLayout()
        self.molecule_input = QComboBox()
        self.molecule_input.addItems(["H2O", "CO2", "CO", "N2", "O2", "CH4", "H2", "NO", "NO2"])
        self.molecule_label = QLabel("Molecule:")
        molecule_layout.addWidget(self.molecule_input)
        molecule_layout.addWidget(self.molecule_label)
        input_layout.addLayout(molecule_layout)

        isotopologue_layout = QHBoxLayout()
        self.isotopologue_input = QComboBox()
        self.isotopologue_input.addItems(["1", "2", "3"])
        self.isotopologue_label = QLabel("Isotopologue")
        isotopologue_layout.addWidget(self.isotopologue_input)
        isotopologue_layout.addWidget(self.isotopologue_label)
        input_layout.addLayout(isotopologue_layout)

        temp_layout = QHBoxLayout()
        self.temp_input = QLineEdit()
        self.temp_label = QLabel("Temperature (K)")
        temp_layout.addWidget(self.temp_input)
        temp_layout.addWidget(self.temp_label)
        input_layout.addLayout(temp_layout)

        pressure_layout = QHBoxLayout()
        self.pressure_input = QLineEdit()
        self.pressure_label = QLabel("Pressure (atm)")
        pressure_layout.addWidget(self.pressure_input)
        pressure_layout.addWidget(self.pressure_label)
        input_layout.addLayout(pressure_layout)

        molar_fraction_layout = QHBoxLayout() 
        self.molar_fraction_input = QLineEdit()
        self.molar_fraction_label = QLabel("Molar Fraction")
        molar_fraction_layout.addWidget(self.molar_fraction_input)
        molar_fraction_layout.addWidget(self.molar_fraction_label)
        input_layout.addLayout(molar_fraction_layout)

        length_layout = QHBoxLayout()
        self.length_input = QLineEdit()
        self.length_label = QLabel("Path Length (cm)")
        length_layout.addWidget(self.length_input)
        length_layout.addWidget(self.length_label)
        input_layout.addLayout(length_layout)

        wavenumber_min_layout = QHBoxLayout()
        self.wavenumber_min_input = QLineEdit()
        self.wavenumber_min_label = QLabel("Minimum Wavenumber (cm-1)")
        wavenumber_min_layout.addWidget(self.wavenumber_min_input)
        wavenumber_min_layout.addWidget(self.wavenumber_min_label)
        input_layout.addLayout(wavenumber_min_layout)

        wavenumber_max_layout = QHBoxLayout()
        self.wavenumber_max_input = QLineEdit()
        self.wavenumber_max_label = QLabel("Maximum Wavenumber (cm-1)")
        wavenumber_max_layout.addWidget(self.wavenumber_max_input)
        wavenumber_max_layout.addWidget(self.wavenumber_max_label)
        input_layout.addLayout(wavenumber_max_layout)

        wavenumber_step_layout = QHBoxLayout()
        self.wavenumber_step_input = QLineEdit()
        self.wavenumber_step_label = QLabel("Wavenumber Step (cm-1)")
        wavenumber_step_layout.addWidget(self.wavenumber_step_input)
        wavenumber_step_layout.addWidget(self.wavenumber_step_label)
        input_layout.addLayout(wavenumber_step_layout)

        method_layout = QHBoxLayout()
        self.method_input = QComboBox()
        self.method_input.addItems(["HT", "V", "L", "D"])
        self.method_label = QLabel("Method")
        method_layout.addWidget(self.method_input)
        method_layout.addWidget(self.method_label)
        input_layout.addLayout(method_layout)

        self.generate_button = QPushButton("Generate Absorption Spectra")
        self.generate_button.clicked.connect(self.generate_absorption_spectra)
        input_layout.addWidget(self.generate_button)

        self.clear_button = QPushButton("Clear Plot")
        self.clear_button.clicked.connect(self.clear_plot)
        input_layout.addWidget(self.clear_button)

        input_group_box.setLayout(input_layout)
        splitter.addWidget(input_group_box)

        # Apply dark background style globally
        #plt.style.use('dark_background')

        # Placeholder for the plot area
        self.plot_canvas = FigureCanvas(Figure())
        self.toolbar = NavigationToolbar(self.plot_canvas, self)
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.plot_canvas)
        plot_frame = QFrame()
        plot_frame.setLayout(plot_layout)
        splitter.addWidget(plot_frame)

        main_layout.addWidget(splitter)
        self.absorption_tab.setLayout(main_layout)

    def init_fitting_tab(self):
        layout = QVBoxLayout()
        label = QLabel("Fitting Data functionality will be here.")
        layout.addWidget(label)
        self.fitting_tab.setLayout(layout)

    def init_lookuptable_tab(self):
        layout = QVBoxLayout()
        label = QLabel("Lookuptable Generation functionality will be here.")
        layout.addWidget(label)
        self.lookuptable_tab.setLayout(layout)

    def generate_absorption_spectra(self):
        molecule = self.molecule_input.currentText()
        # Molecule Id and isotopologue have to be a number
        molecule_id = molecule_id_dict[molecule]
        isotopologue = int(self.isotopologue_input.currentText())
        T = float(self.temp_input.text())
        P = float(self.pressure_input.text())
        molar = float(self.molar_fraction_input.text())
        length = float(self.length_input.text())
        wavenumber_min = float(self.wavenumber_min_input.text())
        wavenumber_max = float(self.wavenumber_max_input.text())
        wavestep = float(self.wavenumber_step_input.text())
        method_ = self.method_input.currentText()
        # Generating the absorption spectra
        data = spectrum(P,T,length,wavenumber_min,wavenumber_max,molecule_id,isotopologue,method_,wavestep,molar)
        # Plotting the absorption spectra in the canvas of the window
        ax = self.plot_canvas.figure.gca()
        plot_spectra(ax,data)
        self.plot_canvas.draw()
        print("Generating absorption spectra...")
    
    def clear_plot(self):
        ax = self.plot_canvas.figure.gca()
        ax.clear()
        self.plot_canvas.draw()
        print("Plot cleared.")

# Run the application
app = QApplication(sys.argv)
window = GUI()
window.show()
sys.exit(app.exec_())