"""
This is the script with the functions to fit the experimental data of absorption spectra of the molecules

It will look for lookuptable first because it is faster than generate the data using HAPI
If the lookuptable is not found, it will generate the data using HAPI and save it in the lookuptable folder for future use
It will take the experimental data and the absorption spectra of the molecule as input and return the fitted data
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lmfit as lm

# from Abs_gen import Absorption

# Importing the variables from the Variables.py script
from Variables import args
from Variables import args_dict