"""
This is the script with the functions to generate the Absorption spectra of the molecules

It will take temperature, pressure, molecule, and other parameters as input and return the absorption spectra of the molecule
The absorption spectra will be generated using the HITRAN database and the HITRAN API

"""
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hapi import *

