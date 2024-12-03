from hapi import *
import numpy as np
import csv


def load_parameters():
    """
    Load parameters for methane absorption calculations.

    Returns
    -------
    dict
        A dictionary containing parameters such as mole fractions, temperatures, pressure, pathlength, etc.
    """
    # Define parameters
    return {
        "molecule_name": "CH4",  # Methane
        "hitran_id": 6,  # HITRAN ID for methane
        "isotopologue_id": 1,  # Default isotopologue
        "wavenumber_start": 3000,  # Start wavenumber (cm⁻¹)
        "wavenumber_end": 3500,  # End wavenumber (cm⁻¹)
        "mole_fractions": [0.1, 0.5, 0.9],  # Test different mole fractions
        "temperatures": [290, 297, 305],  # Test different temperatures
        "pressure": 1.0,  # Pressure in atm
        "pathlength": 10,  # Pathlength in cm
        "spectral_shift": 0.0,  # Spectral shift applied (e.g., cm⁻¹)
        "fit_start": 3050,  # Example fit range start
        "fit_end": 3450,  # Example fit range end
        "wavenumber_array": np.linspace(3000, 3500, 500),  # Higher resolution
        "output_csv_file": "methane_absorption_multiple_conditions.csv",  # Output file
    }


def create_lookup_table(params):
    """
    Generate a lookup table for methane absorption under varying conditions.

    Parameters
    ----------
    params : dict
        Dictionary containing all necessary parameters for calculations.
    """
    # Initialize HAPI database
    db_begin("hapi_data")

    # Fetch methane data from HITRAN
    print(f"Fetching data for {params['molecule_name']} from HITRAN...")
    try:
        fetch(
            params["molecule_name"],
            params["hitran_id"],
            params["isotopologue_id"],
            params["wavenumber_start"],
            params["wavenumber_end"],
        )
        print(f"Data fetched successfully!")
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Create the CSV file for output
    try:
        with open(
            params["output_csv_file"], mode="w", newline="", encoding="utf-8"
        ) as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write headers
            csv_writer.writerow(
                [
                    "Wavenumber (cm⁻¹)",
                    "Absorption Coefficient",
                    "Mole Fraction",
                    "Temperature (K)",
                    "Pressure (atm)",
                    "Pathlength (cm)",
                ]
            )

            # Iterate through combinations of mole fractions and temperatures
            for mole_fraction in params["mole_fractions"]:
                for temperature in params["temperatures"]:
                    print(
                        f"Processing mole fraction {mole_fraction} and temperature {temperature} K..."
                    )
                    diluent_fraction = (
                        1 - mole_fraction
                    )  # Fraction of diluent gas (air)

                    # Compute absorption coefficients using HITRAN data
                    try:
                        nu, absorption_coefficients = absorptionCoefficient_Voigt(
                            [
                                (
                                    params["hitran_id"],
                                    params["isotopologue_id"],
                                    mole_fraction,
                                )
                            ],
                            params["molecule_name"],
                            OmegaGrid=params["wavenumber_array"],
                            Environment={
                                "p": params["pressure"],
                                "T": temperature,
                            },
                            Diluent={
                                "self": mole_fraction,
                                "air": diluent_fraction,
                            },
                        )
                        print(f"Absorption coefficients calculated successfully!")
                    except Exception as e:
                        print(f"Error occurred: {e}")
                        continue

                    # Write spectral data to CSV
                    for wavenumber, absorption in zip(
                        params["wavenumber_array"], absorption_coefficients
                    ):
                        csv_writer.writerow(
                            [
                                wavenumber,
                                f"{absorption:.10e}",
                                mole_fraction,
                                temperature,
                                params["pressure"],
                                params["pathlength"],
                            ]
                        )

            print(f"Results saved to {params['output_csv_file']} (CSV format).")

    except Exception as e:
        print(f"Error occurred while saving to CSV: {e}")


if __name__ == "__main__":
    # Load parameters
    parameters = load_parameters()

    # Create the lookup table
    create_lookup_table(parameters)