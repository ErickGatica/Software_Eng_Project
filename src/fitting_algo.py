"""
This is the script with the functions to fit the experimental data of absorption spectra of the molecules

It will look for lookuptable first because it is faster than generate the data using HAPI
If the lookuptable is not found, it will generate the data using HAPI and save it in the lookuptable folder for future use
It will take the experimental data and the absorption spectra of the molecule as input and return the fitted data
"""
# Libraries

import json
import os
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


def fit_data_from_directory(
        x_wvn,
        transmission,
        config_dir,
        model_setup_func,
        trim_bandwidth_func=pld.trim_bandwidth,
        calc_cepstrum_func=pld.calc_cepstrum,
        weight_func=pld.weight_func,
        plot_intermediate_steps=False,
        verbose=False
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

    Returns
    -------
    Fit : lmfit.Model object
        Fit object containing the model and parameters.
    """
    config_variables = load_config_from_directory(config_dir)
    if not config_variables:
        raise ValueError("No configuration files found or files are empty in the directory.")

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
