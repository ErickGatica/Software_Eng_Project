"""
This is the script with the functions to generate the lookuptable for the absorption spectra of the molecules
It uses the Abs_gen script to generate the absorption spectra of the molecules and then it will save the data in the lookuptable folder
It will take the temperature, pressure, molecule, and other parameters as input and return the lookuptable
# to solve : how we will organize the lookuptable to save the data?
"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hapi import *

