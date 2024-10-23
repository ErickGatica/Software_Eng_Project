"""
With this script we will define the variables that will be used in the Abs_gen.py
 script using argparser

"""

# Libraries

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the command line arguments
parser = argparse.ArgumentParser(description='Abs_spectra')
parser.add_argument('Temperature', type=float, required=True, help='Temperature in Kelvin')
parser.add_argument('Pressure', type=float, required=True, help='Pressure in atm')
parser.add_argument('Molecule', type=str, required=True, help='Molecule name')
parser.add_argument('Concentration', type=float, required=True, help='Molar fraction')
parser.add_argument('min_wavelength', type=float, required=True, help='Minimum wavelength in cm-1') 
parser.add_argument('max_wavelength', type=float, required=True, help='Maximum wavelength in cm-1')
parser.add_argument('resolution', type=float, required=True, help='Resolution in cm-1')
args = parser.parse_args()  

# Save the arguments in a dictionary for easy access

args_dict = {
    'Temperature': args.Temperature,
    'Pressure': args.Pressure,
    'Molecule': args.Molecule,
    'Concentration': args.Concentration,
    'min_wavelength': args.min_wavelength,
    'max_wavelength': args.max_wavelength,
    'resolution': args.resolution
}
