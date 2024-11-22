import pandas as pd
import numpy as np
import itertools
from hapi import db_begin, absorptionCoefficient_Lorentz

# Step 1: Initialize HAPI Database
db_begin('hapi_data')

# Step 2: Define parameter ranges for the lookup table
temperature_range = np.linspace(200, 2000, 10)  # Temperature (K)
mole_fraction_range = np.linspace(0, 1, 10)  # Mole fraction
shift_range = np.linspace(-0.2, 0.2, 5)  # Shift range

# Step 3: Generate the lookup table
lookup_table = []

for temp, mole_frac, shift in itertools.product(temperature_range, mole_fraction_range, shift_range):
    # Example HAPI calculation (replace with specific range/step as needed)
    absorption = absorptionCoefficient_Lorentz(
        'CH4', mole_frac, temp, 1000, 2000, step=0.1
    )

    # Append the result to the lookup table
    lookup_table.append({
        "Temperature (K)": temp,
        "Mole Fraction": mole_frac,
        "Shift": shift,
        "Absorption Coefficients": absorption.tolist()  # Convert array to list for storage
    })

# Step 4: Save the lookup table to a CSV file
lookup_table_df = pd.DataFrame(lookup_table)
lookup_table_df.to_csv('lookup_table.csv', index=False)  # Use the same path as in fitting code
