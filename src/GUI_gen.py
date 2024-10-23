"""
This script generates the GUI for the generation of spectra data, lookuptables and fitting of the experimental data
There is a tab for each of the three functionalities
It uses the pyqt6 library to generate the GUI
"""
# Libraries
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit, QCheckBox
# Importing the variables from the Variables.py script
from Variables import args
from Variables import args_dict

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

    def init_absorption_tab(self):
        layout = QVBoxLayout()

        self.temp_label = QLabel("Temperature (K):")
        self.temp_input = QLineEdit()
        layout.addWidget(self.temp_label)
        layout.addWidget(self.temp_input)

        self.pressure_label = QLabel("Pressure (atm):")
        self.pressure_input = QLineEdit()
        layout.addWidget(self.pressure_label)
        layout.addWidget(self.pressure_input)

        self.molecule_label = QLabel("Molecule:")
        self.molecule_input = QLineEdit()
        layout.addWidget(self.molecule_label)
        layout.addWidget(self.molecule_input)

        self.generate_button = QPushButton("Generate Absorption Spectra")
        self.generate_button.clicked.connect() # connect to the function that 
        # generates the absorption spectra
        layout.addWidget(self.generate_button)

        self.absorption_tab.setLayout(layout)


