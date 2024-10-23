"""
This is the main script that will run the GUI for the generation of spectra data, lookuptables and fitting of the experimental data
"""

# Libraries
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox, QLineEdit, QCheckBox
from Variables import args
from GUI_gen import GUI
from Abs_gen import Absorption
from fitting_algo import fit_data
from lookuptable_gen import generate_lookuptable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
from hapi import *
import os

# Importing the variables from the Variables.py script
from Variables import args
from Variables import args_dict