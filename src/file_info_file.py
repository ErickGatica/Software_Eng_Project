import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pldspectrapy.td_support

# Directory containing .cor files
directory_path = r"D:\FLARE\data\Nico burn meas"
file_list = glob.glob(os.path.join(directory_path, "*.cor"))

# Initialize ranges
wavenumber_min = float("inf")
wavenumber_max = float("-inf")
temperature_min = float("inf")
temperature_max = float("-inf")
mole_fraction_min = float("inf")
mole_fraction_max = float("-inf")

# Process each .cor file
for filepath in file_list:
    try:
        print(f"Processing file: {filepath}")

        # Load the data file
        daq_file = pldspectrapy.open_daq_files(filepath)

        # Extract the spectrum to determine wavenumber range
        x_wvn, transmission = pldspectrapy.td_support.create_spectrum(daq_file)

        # Update wavenumber range
        wavenumber_min = min(wavenumber_min, x_wvn.min())
        wavenumber_max = max(wavenumber_max, x_wvn.max())

        # Extract metadata (temperature, mole fraction) if available
        # Provide defaults if metadata is missing
        temperature = getattr(daq_file, "temperature", None)
        if temperature is not None:
            temperature_min = min(temperature_min, temperature)
            temperature_max = max(temperature_max, temperature)
        else:
            print("Temperature metadata not found. You may need to specify manually.")

        mole_fraction = getattr(daq_file, "mole_fraction", None)
        if mole_fraction is not None:
            mole_fraction_min = min(mole_fraction_min, mole_fraction)
            mole_fraction_max = max(mole_fraction_max, mole_fraction)
        else:
            print("Mole fraction metadata not found. You may need to specify manually.")

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

# Summarize the results
print("\nNecessary Parameter Ranges for Lookup Table:")
print(f"Wavenumber Range: {wavenumber_min:.2f} cm⁻¹ to {wavenumber_max:.2f} cm⁻¹")
if temperature_min < float("inf") and temperature_max > float("-inf"):
    print(f"Temperature Range: {temperature_min:.2f} K to {temperature_max:.2f} K")
else:
    print("Temperature range not found. Specify manually based on experiment.")
if mole_fraction_min < float("inf") and mole_fraction_max > float("-inf"):
    print(f"Mole Fraction Range: {mole_fraction_min:.2f} to {mole_fraction_max:.2f}")
else:
    print("Mole fraction range not found. Specify manually based on experiment.")
