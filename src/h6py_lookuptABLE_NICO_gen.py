from hapi import *
import numpy as np

# Step 1: Initialize HAPI Database
db_begin('hapi_data')  # Creates or uses a local database directory called 'hapi_data'

# Step 2: Fetch Methane Data from HITRAN
molecule_name = 'CH4'  # Methane
hitran_id = 6          # HITRAN ID for methane
isotopologue_id = 1    # Default isotopologue
wavenumber_start = 2900  # Start wavenumber (cm⁻¹)
wavenumber_end = 3000    # End wavenumber (cm⁻¹)

print(f"Fetching data for {molecule_name} from HITRAN...")
fetch(molecule_name, hitran_id, isotopologue_id, wavenumber_start, wavenumber_end)

# Step 3: Define Necessary Variables for Methane
temperature = 296.0  # Standard temperature in Kelvin
mole_fraction = 0.1  # Mole fraction
pressure = 1.0       # Pressure in atm
pathlength = 1.0     # Pathlength in cm
wavenumber_array = np.linspace(wavenumber_start, wavenumber_end, 100)  # Wavenumber grid

# Debugging: Verify input shapes and types
print(f"Molecule: {molecule_name}")
print(f"Mole Fraction: {mole_fraction}, Type: {type(mole_fraction)}")
print(f"Temperature: {temperature}, Type: {type(temperature)}")
print(f"Wavenumber Array: {wavenumber_array}, Shape: {wavenumber_array.shape}")

# Step 4: Compute Absorption Coefficients using HITRAN Data
print(f"Computing absorption coefficients for {molecule_name}...")

try:
    nu, absorption_coefficients = absorptionCoefficient_Voigt(
        [(hitran_id, isotopologue_id, mole_fraction)],  # Molecule definition
        molecule_name,                                 # Database name
        OmegaGrid=wavenumber_array,                   # Wavenumber grid
        Environment={"p": pressure, "T": temperature},  # Environmental conditions
        Diluent={"self": mole_fraction, "air": 1 - mole_fraction},  # Broadening contributions
    )
    print(f"Absorption coefficients calculated successfully!")
except Exception as e:
    print(f"Error occurred: {e}")
    exit()

# Step 5: Output Results to File with UTF-8 Encoding
output_file = "enhanced_methane_absorption_output.txt"
try:
    with open(output_file, 'w', encoding='utf-8') as f:  # Explicitly set encoding
        # Write header with superscript characters
        f.write("Wavenumber (cm⁻¹), Absorption Coefficient\n")
        for wavenumber, absorption in zip(wavenumber_array, absorption_coefficients):
            f.write(f"{wavenumber}, {absorption}\n")
    print(f"Results saved to {output_file} (UTF-8 encoded).")

except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
    print("Falling back to ASCII-compatible output...")
    output_file = "enhanced_methane_absorption_output_ascii.txt"
    with open(output_file, 'w', encoding='ascii') as f:  # ASCII fallback
        # Replace superscript characters with ASCII equivalents
        f.write("Wavenumber (cm^-1), Absorption Coefficient\n")
        for wavenumber, absorption in zip(wavenumber_array, absorption_coefficients):
            f.write(f"{wavenumber}, {absorption}\n")
    print(f"Results saved to {output_file} (ASCII fallback).")
