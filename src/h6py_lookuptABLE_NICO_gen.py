from hapi import *
import numpy as np
import csv

# Step 1: Initialize HAPI Database
db_begin('hapi_data')  # Creates or uses a local database directory called 'hapi_data'

# Step 2: Fetch Methane Data from HITRAN
molecule_name = 'CH4'  # Methane
hitran_id = 6  # HITRAN ID for methane
isotopologue_id = 1  # Default isotopologue
wavenumber_start = 3000  # Start wavenumber (cm⁻¹)
wavenumber_end = 3500  # End wavenumber (cm⁻¹)

print(f"Fetching data for {molecule_name} from HITRAN...")
fetch(molecule_name, hitran_id, isotopologue_id, wavenumber_start, wavenumber_end)

# Step 3: Define Necessary Variables for Methane
temperature = 297  # Room temperature in Kelvin
pressure = 1.0  # Pressure in atm
pathlength = 10  # Pathlength in cm
mole_fraction = 0.5  # Mole fraction (50% methane)
wavenumber_array = np.linspace(wavenumber_start, wavenumber_end, 500)  # Higher resolution

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
        molecule_name,  # Database name
        OmegaGrid=wavenumber_array,  # Wavenumber grid
        Environment={"p": pressure, "T": temperature},  # Environmental conditions
        Diluent={"self": mole_fraction, "air": 1 - mole_fraction},  # Broadening contributions
    )
    print(f"Absorption coefficients calculated successfully!")
except Exception as e:
    print(f"Error occurred: {e}")
    exit()

# Step 5: Calculate Mole Fraction from Absorption Coefficients
print("Calculating mole fraction based on HAPI data...")
average_absorption = np.mean(absorption_coefficients)  # Average coefficient for simplicity
calculated_mole_fraction = (
        average_absorption * pathlength * temperature / pressure
)

# Step 6: Save Results to CSV
output_csv_file = "methane_absorption_data.csv"
try:
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write headers
        csv_writer.writerow(["Wavenumber (cm⁻¹)", "Absorption Coefficient", "Temperature (K)", "Pressure (atm)"])

        # Write data rows
        for wavenumber, absorption in zip(wavenumber_array, absorption_coefficients):
            csv_writer.writerow([wavenumber, f"{absorption:.10e}", temperature, pressure])

        # Write summary information
        csv_writer.writerow([])
        csv_writer.writerow(["Calculated Mole Fraction", f"{calculated_mole_fraction:.10e}"])

    print(f"Results saved to {output_csv_file} (CSV format).")

except Exception as e:
    print(f"Error occurred while saving to CSV: {e}")
