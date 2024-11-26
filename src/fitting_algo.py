import h5py
import numpy as np
import os
from tqdm import tqdm
from lmfit import Model
import pldspectrapy as pld  # Assuming pldspectrapy package provides necessary spectral analysis tools


def load_lookup_table_hdf5(hdf5_path):
    """
    Loads the lookup table from an HDF5 file.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.

    Returns
    -------
    hdf5_file : h5py.File or None
        The opened HDF5 file handle or None if the file doesn't exist.
    """
    if os.path.exists(hdf5_path):
        return h5py.File(hdf5_path, 'a')  # Open in append mode
    else:
        return None


def get_spectral_data_from_hdf5_or_generate(hdf5_file, config, x_wvn):
    """
    Retrieves spectral data from the HDF5 lookup table if available; otherwise, generates it.

    Parameters
    ----------
    hdf5_file : h5py.File
        The HDF5 file handle for the lookup table.
    config : dict
        Configuration dictionary with fitting parameters.
    x_wvn : numpy array
        Wavenumber array.

    Returns
    -------
    spectral_data : numpy array
        The spectral data for the given conditions.
    """
    temperature = config["fitting"]["temperature"]
    mole_fraction = config["fitting"]["molefraction"]
    shift = config["fitting"]["shift"]

    if hdf5_file is not None:
        # Construct indices to look up data
        temp_idx = np.argmin(np.abs(hdf5_file['Temperature'][:] - temperature))
        mole_idx = np.argmin(np.abs(hdf5_file['MoleFraction'][:] - mole_fraction))
        shift_idx = np.argmin(np.abs(hdf5_file['Shift'][:] - shift))

        # Check if the data exists
        try:
            spectral_data = hdf5_file['AbsorptionCoefficients'][temp_idx, mole_idx, shift_idx, :]
            if np.all(np.isfinite(spectral_data)):
                return spectral_data
        except KeyError:
            pass

    # If not in the lookup table, generate the data using HAPI or other tools
    print(f"No matching data found in HDF5. Generating for T={temperature}, mole_fraction={mole_fraction}, shift={shift}...")
    spectral_data = pld.generate_spectral_data(x_wvn, temperature, mole_fraction, shift)

    # Save the new data into the HDF5 table
    if hdf5_file is not None:
        temp_idx = np.argmin(np.abs(hdf5_file['Temperature'][:] - temperature))
        mole_idx = np.argmin(np.abs(hdf5_file['MoleFraction'][:] - mole_fraction))
        shift_idx = np.argmin(np.abs(hdf5_file['Shift'][:] - shift))
        hdf5_file['AbsorptionCoefficients'][temp_idx, mole_idx, shift_idx, :] = spectral_data

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
        lookup_table_path="lookup_table.h5"
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
        Path to the HDF5 lookup table file.

    Returns
    -------
    Fit : lmfit.Model object
        Fit object containing the model and parameters.
    """
    config_variables = load_config_from_directory(config_dir)
    if not config_variables:
        raise ValueError("No configuration files found or files are empty in the directory.")

    hdf5_file = load_lookup_table_hdf5(lookup_table_path)
    spectral_data = get_spectral_data_from_hdf5_or_generate(hdf5_file, config_variables, x_wvn)

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

    if hdf5_file is not None:
        hdf5_file.close()

    return Fit
