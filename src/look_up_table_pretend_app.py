import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# Load the lookup table
lookup_table = pd.read_pickle('lookup_table.pkl')

# Function to retrieve absorption coefficients
def get_absorption_coefficients(temp, mole_frac, shift):
    # Interpolate absorption coefficients from the lookup table
    points = lookup_table[['Temperature (K)', 'Mole Fraction', 'Shift']].values
    values = lookup_table['Absorption Coefficients'].tolist()
    return griddata(points, values, (temp, mole_frac, shift), method='linear')

# Example usage
temp = 300
mole_frac = 0.05
shift = 0.1
absorption_coefficients = get_absorption_coefficients(temp, mole_frac, shift)
print(absorption_coefficients)
