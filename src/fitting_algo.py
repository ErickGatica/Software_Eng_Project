# Libraries
import json
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from lmfit import Model
import pldspectrapy as pld  # Assuming pldspectrapy package provides necessary spectral analysis tools


def load_config_from_directory(config_dir):
    """
    Loads configuration parameters from a specified directory.

    Parameters
    ----------
    config_dir : str
        Directory path where configuration files are stored.

    Returns
    -------
    config : dict
        Dictionary containing configuration parameters loaded from the files.
    """
    config = {}
    for filename in os.listdir(config_dir):
        if filename.endswith(".json"):
            with open(os.path.join(config_dir, filename), 'r') as f:
                config.update(json.load(f))
    return config


def setup_model_with_config(config_variables, model_setup_func):
    """
    Set up the model and parameters based on configuration variables.

    Parameters
    ----------
    config_variables : dict
        Dictionary with fitting configuration parameters.
    model_setup_func : callable
        Function to set up models using the configuration.

    Returns
    -------
    model : lmfit.Model
        The initialized model for fitting.
    parameters : lmfit.Parameters
        The model parameters.
    """
    return model_setup_func(config_variables)


def load_lookup_table(table_path):
    """
    Loads the lookup table if it exists.

    Parameters
    ----------
    table_path : str
        Path to the lookup table file.

    Returns
    -------
    lookup_table : pandas.DataFrame
        The loaded lookup table.
    """
    if os.path.exists(table_path):
        return pd.read_csv(table_path)
    else:
        return None


def get_spectral_data_from_lookup_or_generate(lookup_table, config, x_wvn, transmission):
    """
    Retrieves spectral data from the lookup table if available; otherwise, generates it.

    Parameters
    ----------
    lookup_table : pandas.DataFrame
        The lookup table containing precomputed spectral data.
    config : dict
        Configuration dictionary with fitting parameters.
    x_wvn : numpy array
        Wavenumber array.
    transmission : numpy array
        Transmission array.

    Returns
    -------
    spectral_data : numpy array
        The spectral data for the given conditions.
    """
    temperature = config["fitting"]["temperature"]
    mole_fraction = config["fitting"]["molefraction"]
    shift = config["fitting"]["shift"]

    # Check if the data exists in the lookup table
    if lookup_table is not None:
        match = lookup_table[
            (lookup_table["Temperature (K)"] == temperature) &
            (lookup_table["Mole Fraction"] == mole_fraction) &
            (lookup_table["Shift"] == shift)
        ]

        if not match.empty:
            return np.array(match["Absorption Coefficients"].iloc[0])

    # If not in the lookup table, generate the data using HAPI or other tools
    print(f"No matching data found in lookup table. Generating for T={temperature}, mole_fraction={mole_fraction}, shift={shift}...")
    spectral_data = pld.generate_spectral_data(x_wvn, temperature, mole_fraction, shift)

    # Save the new data to the lookup table
    new_entry = {
        "Temperature (K)": temperature,
        "Mole Fraction": mole_fraction,
        "Shift": shift,
        "Absorption Coefficients": spectral_data.tolist(),
    }
    if lookup_table is not None:
        lookup_table = pd.concat([lookup_table, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        lookup_table = pd.DataFrame([new_entry])
    lookup_table.to_csv('lookup_table.csv', index=False)

    return spectral_data


def fit_data_from_directory(
        x_wvn,
        transmission,
        config_dir,
        model_setup_func,
        trim_bandwidth_func=pld.trim_bandwidth,
        calc_cepstrum_func=pld.calc_cepstrum,
        weight_func=pld.weight_func,
        plot_intermediate_steps=False,
        verbose=False,
        lookup_table_path="lookup_table.csv"
):
    """
    Fits spectrum to a model using lmfit and parameters from a directory of configuration files.

    Parameters
    ----------
    x_wvn : numpy array
        Wavenumber array.
    transmission : numpy array
        Transmission array.
    config_dir : str
        Path to the directory containing configuration files.
    model_setup_func : callable
        Function to set up the model.
    trim_bandwidth_func : callable, optional
        Function for trimming the bandwidth, defaulting to pldspectrapy.
    calc_cepstrum_func : callable, optional
        Function to calculate cepstrum.
    weight_func : callable, optional
        Function to calculate weights.
    plot_intermediate_steps : bool, optional
        If True, plots intermediate steps of fitting.
    verbose : bool, optional
        If True, prints parameter information.
    lookup_table_path : str, optional
        Path to the lookup table file.

    Returns
    -------
    Fit : lmfit.Model object
        Fit object containing the model and parameters.
    """
    config_variables = load_config_from_directory(config_dir)
    if not config_variables:
        raise ValueError("No configuration files found or files are empty in the directory.")

    lookup_table = load_lookup_table(lookup_table_path)
    spectral_data = get_spectral_data_from_lookup_or_generate(lookup_table, config_variables, x_wvn, transmission)

    pbar = tqdm(total=None, desc="Fitting progress")

    def callback(params, iter, resid, *args, **kwargs):
        pbar.update(1)

    x_wvn_trimmed, transmission_trimmed = trim_bandwidth_func(
        x_wvn, transmission, config_variables["fitting"]["band_fit"]
    )
    y_td = calc_cepstrum_func(transmission_trimmed)

    weight_array = weight_func(
        len(x_wvn_trimmed),
        config_variables["fitting"]["baseline"],
        config_variables["fitting"]["etalons"],
    )

    model, parameters = setup_model_with_config(config_variables, model_setup_func)
    if verbose:
        parameters.pretty_print()

    Fit = model.fit(
        y_td,
        xx=x_wvn_trimmed,
        params=parameters,
        weights=weight_array,
        iter_cb=callback,
    )

    pbar.close()

    if plot_intermediate_steps:
        generate_tutorial_fit_plots(x_wvn, transmission, config_variables, Fit)

    return Fit
