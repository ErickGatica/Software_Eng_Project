import time
import warnings
import os
import glob
import matplotlib.pyplot as plt
from scipy import interpolate as intrp
import allantools as ad
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib as mpl
import pldspectrapy.td_support
from pldspectrapy import misc_tools, fit_data
from pldspectrapy.config_handling import load_config_json

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configure plot formatting
def configure_plots():
    mpl.rcParams["backend"] = "TkAgg"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["font.family"] = "Arial"
    plt.rcParams.update({"figure.autolayout": True, "lines.linewidth": 0.65})
    plt.rcParams.update({"mathtext.default": "regular"})

# Load configuration file
def load_configuration(config_path):
    return load_config_json(config_path)

# Initialize the HAPI database
def initialize_hapi_db(config_variables):
    if config_variables["fitting"]["simulation_backend"] != "gaas":
        pldspectrapy.td_support.initialize_hapi_db(
            config_variables["input"]["linelist_path"]
        )

# Process a single file
def process_file(filepath, config_variables):
    try:
        filename = os.path.basename(filepath)
        print(f"Processing file: {filepath}")

        # Update configuration with the current file path
        config_variables["input"]["filename"] = filepath

        # Load the data file
        daq_file = pldspectrapy.open_daq_files(filepath)

        # Prepare the file for processing
        daq_file.prep_for_processing(config_variables)

        # Generate spectrum
        x_wvn, transmission = pldspectrapy.td_support.create_spectrum(
            daq_file, config_variables
        )

        # Perform fitting or other analysis
        Fit = fit_data(x_wvn, transmission, config_variables)

        # Generate output and save results
        results_df = pldspectrapy.config_handling.generate_output_and_save(
            Fit, config_variables
        )

        # Optionally, plot results
        if config_variables["input"]["fit_plots"]:
            plot_results(Fit, config_variables)

        # Print a report
        print(Fit.fit_report())

    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

# Plot results
def plot_results(Fit, config_variables):
    pldspectrapy.plotting_tools.plot_fit_td(Fit, config_variables)
    pldspectrapy.plotting_tools.plot_fit_freq(Fit, config_variables)

# Main processing function
def main():
    configure_plots()

    start_time = time.time()
    os.environ["GAAS_OCL_DEVICE"] = "0"

    # Load configuration
    config_path = os.path.join("C:\\git\\flare", "flare_config_1mod.json5")
    config_variables = load_configuration(config_path)

    # Initialize HAPI database if necessary
    initialize_hapi_db(config_variables)

    # Directory and file handling
    directory_path = r"D:\FLARE\data\Nico burn meas"
    file_list = glob.glob(os.path.join(directory_path, "*.cor"))

    # Process files
    n = min(10, len(file_list))
    for i in range(n):
        process_file(file_list[i], config_variables)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

# Entry point
if __name__ == "__main__":
    main()
