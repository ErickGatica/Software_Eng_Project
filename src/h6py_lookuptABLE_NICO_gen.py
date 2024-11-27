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

# Step 3: Define Necessary Variables
mole_fractions = [0.1, 0.5, 0.9]  # Test different mole fractions
temperatures = [290, 297, 305]  # Test different temperatures
pressure = 1.0  # Pressure in atm
pathlength = 10  # Pathlength in cm
spectral_shift = 0.0  # Spectral shift applied (e.g., cm⁻¹)
fit_start = wavenumber_start + 50  # Example fit range
fit_end = wavenumber_end - 50
wavenumber_array = np.linspace(wavenumber_start, wavenumber_end, 500)  # Higher resolution

# Step 4: Prepare CSV File for Output
output_csv_file = "methane_absorption_multiple_conditions.csv"

try:
    with open(output_csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write headers
        csv_writer.writerow(["Wavenumber (cm⁻¹)", "Absorption Coefficient", "Mole Fraction", "Temperature (K)", "Pressure (atm)", "Pathlength (cm)"])

        # Iterate through combinations of mole fractions and temperatures
        for mole_fraction in mole_fractions:
            for temperature in temperatures:
                print(f"Processing mole fraction {mole_fraction} and temperature {temperature} K...")
                diluent_fraction = 1 - mole_fraction  # Fraction of diluent gas (air)

                # Step 5: Compute Absorption Coefficients using HITRAN Data
                try:
                    nu, absorption_coefficients = absorptionCoefficient_Voigt(
                        [(hitran_id, isotopologue_id, mole_fraction)],  # Molecule definition
                        molecule_name,  # Database name
                        OmegaGrid=wavenumber_array,  # Wavenumber grid
                        Environment={"p": pressure, "T": temperature},  # Environmental conditions
                        Diluent={"self": mole_fraction, "air": diluent_fraction},  # Broadening contributions
                    )
                    print(f"Absorption coefficients calculated successfully!")
                except Exception as e:
                    print(f"Error occurred: {e}")
                    continue

                # Step 6: Write spectral data to CSV
                for wavenumber, absorption in zip(wavenumber_array, absorption_coefficients):
                    csv_writer.writerow([wavenumber, f"{absorption:.10e}", mole_fraction, temperature, pressure, pathlength])

        print(f"Results saved to {output_csv_file} (CSV format).")

except Exception as e:
    print(f"Error occurred while saving to CSV: {e}")
