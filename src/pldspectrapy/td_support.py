# -*- coding: utf-8 -*-
"""
Universal time-domain codes.

For implementation into pldspectrapy.
Can handle multispecies fitting, each with their own full path characteristics.
You can apply a constraint to match, say, pathlength, pressure, temperature of each.

Created on Tue Nov  5 13:30:40 2019

@author: Nate Malarich
"""
import os, sys

# built-in modules
import numpy as np
import matplotlib.pyplot as plt

from lmfit import Model
from tqdm import tqdm

import pldspectrapy as pld

from pldspectrapy.constants import MOLECULE_IDS

from . import pldhapi
from . import model_creation
from .misc_tools import get_hitran_molecule_id
from .igtools import avg_igs_by_idx

# TODO: Is this overwriting the pldhapi import above?
import hapi
#import hapi2
from hapi2.opacity.lbl.numba import (
    absorptionCoefficient_Voigt as absorptionCoefficient_Voigt_hapi2,
)
from hapi import volumeConcentration

# from radis import SpectrumFactory # Comment out for debugging to prevent hanging on hdf5 stuff from radis
# from constants import SPEED_OF_LIGHT
# import gaas_ocl as gs

# This is copied from model_creation.py
# TODO Create global or CONSTANT?
exclude_names = ["global", "db_name"]


def largest_prime_factor(n):
    """
    Want 2 * (x_stop - x_start - 1) to have small largest_prime_factor
    """
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n


def bandwidth_select_td(x_array, band_fit, max_prime_factor=500):
    # TODO: replace with scipy.fft.next_fast_len
    # TODO: look into automatic etalon phase matching for narrow etalons
    """
    Tweak bandwidth selection for swift time-domain fitting.

    Time-domain fit_td does inverse FFT for each nonlinear least-squares iteration,
    and speed of FFT goes with maximum prime factor.

    INPUTS:
        x_array = x-axis for measurement transmission spectrum
        band_fit = [start_frequency, stop_frequency]
    """
    x_start = np.argmin(np.abs(x_array - band_fit[0]))
    x_stop = np.argmin(np.abs(x_array - band_fit[1]))

    len_td = 2 * (np.abs(x_stop - x_start) - 1)  # np.fft.irfft operation
    # len_td = np.abs(x_stop - x_start)  # np.fft.irfft operation
    prime_factor = largest_prime_factor(len_td)

    while prime_factor > max_prime_factor:
        x_stop -= 1
        len_td = 2 * (np.abs(x_stop - x_start) - 1)
        # len_td = np.abs(x_stop - x_start)
        prime_factor = largest_prime_factor(len_td)
    return x_start, x_stop


# Write function to shrink bandwidth based on start_pnt, stop_pnt
def bandwidth_trim(x_wvn_full, trans_full, start_pnt, stop_pnt, band_fit):
    """
    Use the output of bandwidth_select_td to trim x_array and transmission.

    Args:
        x_wvn_full (array) :The x-axis for the measurement transmission spectrum.
        Trans_full (array) :The transmission spectrum.
        Start_pnt (int) :The index of the start of the transmission spectrum.
        Stop_pnt (int) :The index of the end of the transmission spectrum.

    Returns:
        x_wvn_trimmed (array) :The trimmed x-axis.
        Trans_trimmed (array) :The trimmed transmission spectrum.
    """
    if start_pnt < stop_pnt:
        # Normal setup
        trans_trimmed = trans_full[start_pnt:stop_pnt]
        x_wvn_trimmed = x_wvn_full[start_pnt:stop_pnt]
    else:
        # DCS in 0.5-1.0 portion of Nyquist window, need to flip x-axis to fit
        # print(f"Length of raw transmission: {len(trans_full)}")
        trans_flipped = trans_full[(int(len(trans_full) / 2)) :: -1]
        # print(f"Length of flipped transmission {len(trans_flipped)}")
        x_wvn_flipped = x_wvn_full[::-1]
        start_pnt, stop_pnt = bandwidth_select_td(
            x_wvn_flipped, band_fit, max_prime_factor=1000
        )
        x_wvn_trimmed = x_wvn_flipped[start_pnt:stop_pnt]
        trans_trimmed = trans_flipped[start_pnt:stop_pnt]

    return x_wvn_trimmed, trans_trimmed


def calc_cepstrum(trans):
    """
    Calculate cepstrum from transmission spectrum

    Parameters
    __________
    trans: array
        Transmission Spectrum (usually trimmed to bandfit of interest)

    Returns
    -------
    y_td: array
        Cepstrum of the transmission spectrum

    """

    y_td = np.fft.irfft(-np.log(trans))
    return y_td


def bandwidth_select_td_grow(x_array, band_fit, max_prime_factor=500):
    """
    Try to mimic behavior of scipy.fft.next_fast_len by growing the range.
    Author: Eli Miller

    INPUTS:
        x_array = x-axis for measurement transmission spectrum
        band_fit = [start_frequency, stop_frequency]
    """
    x_start = np.argmin(np.abs(x_array - band_fit[0]))
    x_stop = np.argmin(np.abs(x_array - band_fit[1]))

    len_td = 2 * (np.abs(x_stop - x_start) - 1)  # np.fft.irfft operation
    # len_td = np.abs(x_stop - x_start)  # np.fft.irfft operation
    prime_factor = largest_prime_factor(len_td)

    while prime_factor > max_prime_factor:
        x_stop += 1
        len_td = 2 * (np.abs(x_stop - x_start) - 1)
        # len_td = np.abs(x_stop - x_start)
        prime_factor = largest_prime_factor(len_td)
    return x_start, x_stop


def weight_func(spectrum_length, baseline_length, etalons=None):
    """
    Calculates time-domain weighting function, set to 0 over selected baseline, etalon range
    INPUTS:
        spectrum_length = length of frequency-domain spectrum
        baseline_length = number of points at beginning to attribute to baseline
        etalon_ranges = list of [start_point, stop_point] time-domain points for etalon spikes
    """

    if etalons is None:
        etalons = []

    weight_array = np.ones(2 * (spectrum_length - 1))

    if baseline_length > 0:
        weight_array[0:baseline_length] = 0
        weight_array[-baseline_length:] = 0
    elif baseline_length == 0:
        weight_array[0] = 0
    else:
        raise ValueError("Baseline length must be positive or zero.")

    for etalon_range in etalons:
        weight_array[etalon_range[0] : etalon_range[1]] = 0
        weight_array[-etalon_range[1] : -etalon_range[0]] = 0

    return weight_array


"""
Wrapper codes for producing absorption models in time-domain.
To be called using lmfit nonlinear least-squares
"""


def spectra_single(
    xx,
    mol_id,
    iso,
    molefraction,
    pressure,
    temperature,
    pathlength,
    shift,
    db_name=None,
    flip_spectrum=False,
):
    """
    Spectrum calculation for single absorption species.

    Parameters
    ----------
    xx : array
        x-axis for spectrum
    mol_id : int
        Hitran integer for molecule
    iso : int
        Hitran integer for isotope
    molefraction : float
        Mole fraction
    pressure : float
        Pressure in atm
    temperature : float
        Temperature in K
    pathlength : float
        Pathlength in cm
    shift : float
        Shift in cm-1
    db_name : str
        Name of linelist
    db_names : dict, optional
        Dictionary of linelist names. The default is None.
    flip_spectrum : bool, optional
        Flip spectrum. The default is False.

    Returns
    -------
    absorp : array
        Absorption spectrum. This is actually the time-domain spectrum (cepstrum) to use in the fit.

    #TODO: fix this to actually return the absorption spectrum or change the naming to reflect that it's the cepstrum

    Spectrum calculation for adding multiple models with composite model.

    See lmfit model page on prefix, parameter hints, composite models.
    """
    if db_name is None:
        raise ValueError("db_name must be specified")

    nu, coef = pldhapi.absorptionCoefficient_Voigt(
        ((int(mol_id), int(iso), molefraction),),
        db_name,
        HITRAN_units=False,
        OmegaGrid=xx + shift,
        Environment={"p": pressure, "T": temperature},
        Diluent={"self": molefraction, "air": (1 - molefraction)},
    )
    coef *= hapi.abundance(int(mol_id), int(iso))  # assume natural abundance

    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)
    return absorp


def spectra_single_hapi2(
    xx,
    mol_id,
    iso,
    molefraction,
    pressure,
    temperature,
    pathlength,
    shift,
    db_name=None,
    flip_spectrum=False,
):
    """
    Spectrum calculation for single absorption species.

    Parameters
    ----------
    xx : array
        x-axis for spectrum
    mol_id : int
        Hitran integer for molecule
    iso : int
        Hitran integer for isotope
    molefraction : float
        Mole fraction
    pressure : float
        Pressure in atm
    temperature : float
        Temperature in K
    pathlength : float
        Pathlength in cm
    shift : float
        Shift in cm-1
    db_name : str
        Name of linelist
    db_names : dict, optional
        Dictionary of linelist names. The default is None.
    flip_spectrum : bool, optional
        Flip spectrum. The default is False.

    Returns
    -------
    absorp : array
        Absorption spectrum. This is actually the time-domain spectrum (cepstrum) to use in the fit.

    #TODO: fix this to actually return the absorption spectrum or change the naming to reflect that it's the cepstrum

    Spectrum calculation for adding multiple models with composite model.

    See lmfit model page on prefix, parameter hints, composite models.
    """
    if db_name is None:
        raise ValueError("db_name must be specified")

    nu, coef = absorptionCoefficient_Voigt_hapi2(
        ((int(mol_id), int(iso), molefraction),),
        db_name,
        HITRAN_units=True,
        WavenumberGrid=xx + shift,
        Environment={"p": pressure, "T": temperature},
        Diluent={"self": molefraction, "air": (1 - molefraction)},
    )

    # coef *= hapi.abundance(int(mol_id), int(iso))  # It seems like hapi2 already accounts for abundance

    coef *= (
        volumeConcentration(pressure, temperature) * molefraction
    )  # This is to convert from HITRAN_units to cm-1. This is not yet implemented inside hapi2

    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)
    return absorp


def spectra_single_gaas(
    xx,
    temperature,
    pressure,
    molefraction,
    mol_id,
    iso,
    pathlength,
    shift,
    absDB_dict=None,
    db_name=None,
    flip_spectrum=False,
    gaas_key=None,
):
    """

    Parameters
    ----------
    temperature : float
    pressure : float
    molefraction : float
    xx : array
    iso : int
    absDB : array
    pathlength : float
    shift : float
    db_name : str`
    flip_spectrum : bool

    Returns
    -------
    absorp : array
        Absorption spectrum. This is actually the time-domain spectrum (cepstrum) to use in the fit.

    """
    # #TODO: fix this to actually return the absorption spectrum or change the naming to reflect that it's the cepstrum
    #
    # Spectrum calculation for adding multiple models with composite model.
    #
    # See lmfit model page on prefix, parameter hints, composite models.

    startWavenum = xx[0]
    endWavenum = xx[len(xx) - 1]
    wavenumStep = (endWavenum - startWavenum) / (len(xx) - 1)

    mol_name = list(MOLECULE_IDS.keys())[get_hitran_molecule_id(db_name) - 1]

    absDB = absDB_dict[gaas_key]

    nu, coef = gs.simVoigt(
        temperature,
        pressure,
        molefraction,
        wavenumStep,
        startWavenum + shift,
        endWavenum + shift,
        mol_name,
        iso,
        absDB,
        gs.get_tips_calc(mol_name, iso),
    )

    coef *= hapi.abundance(int(mol_id), int(iso))  # assume natural abundance

    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)

    return absorp


def spectra_single_lmfit(prefix="", sd=False):
    """
    Set up lmfit model with function hints for single absorption species
    """
    if sd:
        mod = Model(spectra_sd, prefix=prefix)
    else:
        mod = Model(spectra_single, prefix=prefix)
    mod.set_param_hint("mol_id", vary=False)
    mod.set_param_hint("iso", vary=False)
    mod.set_param_hint("pressure", min=0)
    mod.set_param_hint("temperature", min=0)
    mod.set_param_hint("pathlength", min=0)
    mod.set_param_hint("molefraction", min=0, max=1)
    mod.set_param_hint("shift", value=0, min=-0.2, max=0.2)
    pars = mod.make_params()
    # let's set up some default thermodynamics
    pars[prefix + "mol_id"].value = 1
    pars[prefix + "iso"].value = 1
    pars[prefix + "pressure"].value = 640 / 760
    pars[prefix + "temperature"].value = 296
    pars[prefix + "pathlength"].value = 100
    pars[prefix + "molefraction"].value = 0.01

    return mod, pars


def spectra_single_radis(
    xx,
    mol_id,
    iso,
    molefraction,
    pressure,
    temperature,
    pathlength,
    shift,
    name="H2O",
    flip_spectrum=False,
    sf_to_use=None,
):
    """Wrapper to use radis SpectrumFactory with existing code.

    Parameters
    ----------
    xx : array
        x-axis for spectrum
    mol_id : int
        Hitran integer for molecule
    iso : int
        Hitran integer for isotope
    molefraction : float
        Mole fraction
    pressure : float
        Pressure in atm
    temperature : float
        Temperature in K
    pathlength : float
        Pathlength in cm
    shift : float
        Shift in cm-1
    name : str, optional
        Name of linelist. The default is "H2O".
    flip_spectrum : bool, optional
        Flip spectrum. The default is False.

    Returns
    -------
    absorp : array
        Absorption spectrum.

    """

    # Figure out wstep from xx
    # TODO: FIX THIS to be more elegant. Right now, it's just a hack.
    # Create dictionary of Hitran molecule IDs to the strings that radis uses

    # Convert pressure from atm to bar
    pressure *= 1.01325

    mol_dict = {1: "H2O", 2: "CO2", 3: "O3", 4: "N2O", 5: "CO", 6: "CH4"}

    wstep = np.diff(xx)[0]  # Assume uniform spacing in xx
    # Create SpectrumFactory object
    # TODO: move this out of the function to avoid reloading the linelist every time. Pass it in as an argument.

    if sf_to_use is None:
        sf = SpectrumFactory(
            molecule=mol_dict[mol_id],
            wavenum_min=min(xx),
            wavenum_max=max(xx),
            isotope=[iso],
            wstep=wstep,
            verbose=2,
            cutoff=0,
            truncation=300,
            # diluent={'air': 1 - molefraction, 'self': molefraction},
            # broadening_method='voigt',
            # optimization=None
        )
    else:
        sf = sf_to_use

    # Calculate absorption spectrum
    if name == "CH4_HIT08":
        pass
    else:
        name = "HITRAN-" + name

    # TODO: Don't load the linelist every time
    sf.load_databank(name)
    spectrum = sf.eq_spectrum(
        Tgas=temperature,
        path_length=pathlength,
        pressure=pressure,
        mole_fraction=molefraction,
    )

    # Extract wavelength and absorption information
    x_radis, absorp = spectrum.get("absorbance")

    # Wavelength shift isn't built into radis, so we have to do it manually
    x_radis += shift

    # interpolate our spectrum to the same x-axis as the data

    absorp = np.interp(xx, x_radis, absorp)

    # Flip spectrum if needed
    if flip_spectrum:
        absorp = absorp[::-1]

    # return absorp
    # This returns the time-domain spectrum (cepstrum) to use in the fit
    return np.fft.irfft(absorp)


# TODO: Synchronize naming between spectra_single, spectra_sd, spectra_single_radis etc.
def spectra_sd(
    xx,
    mol_id,
    iso,
    molefraction,
    pressure,
    temperature,
    pathlength,
    shift,
    name="H2O",
    flip_spectrum=False,
):
    """
    Spectrum calculation for adding multiple models with composite model.

    See lmfit model page on prefix, parameter hints, composite models.

    INPUTS:
        xx -> wavenumber array (cm-1)
        name -> name of file (no extension) to pull linelist
        mol_id -> Hitran integer for molecule
        iso -> Hitran integer for isotope
        molefraction
        pressure -> (atmospheres)
        temperature -> kelvin
        pathlength (centimeters)
        shift -> (cm-1) calculation relative to Hitran
        flip_spectrum -> set to True if Nyquist window is 0.5-1.0

    """

    nu, coef = pldhapi.absorptionCoefficient_SDVoigt(
        ((int(mol_id), int(iso), molefraction),),
        name,
        HITRAN_units=False,
        OmegaGrid=xx + shift,
        Environment={"p": pressure, "T": temperature},
        Diluent={"self": molefraction, "air": (1 - molefraction)},
    )
    coef *= hapi.abundance(int(mol_id), int(iso))  # assume natural abundance
    if flip_spectrum:
        absorp = np.fft.irfft(coef[::-1] * pathlength)
    else:
        absorp = np.fft.irfft(coef * pathlength)
    return absorp


def spectra_scale_model(xx, scalefactor, scale_key=None, scale_models=None):
    """
    Model function for completing a scale-model fit
    Parameters
    ----------
    xx: numpy array
        Wavenumber axis for data -> this parameter is not used in this function but
        had to be included to maintain compatibility with other model functions
    scalefactor: float
        Scalefactor that is applied to the cepstrum during fitting
    scale_key: int
        Index key for extracting the correct model from scale_models
    scale_models: numpy array
        Matrix of base models for each molecule being fit

    Returns
    -------
    scaled_cepstrum: base model * scalefactor
    """

    td_model = scale_models[scale_key]

    scaled_cepstrum = td_model * scalefactor

    return scaled_cepstrum


"""
Tools for plotting results and baseline removal.
"""


def lmfit_uc(Fit, str_param):
    """
    Get statistical fitting uncertainty of some fit_td parameter named str_param
    INPUTS:
        Fit = lmfit Model result object (Fit = mod.fit_td(...))
        str_param = name of parameter to extract
    warning: some fits are unstable and cannot calculate statistical uncertainties
    """
    fit_report = Fit.fit_report()
    for line in fit_report.split("\n"):
        if (str_param + ":") in line:
            foo = line.split()
            fit_value = float(foo[1])

            try:
                fit_uc = float(foo[3])
            except:
                fit_uc = np.nan

    return fit_uc


def plot_fit(x_data, Fit, tx_title=True):
    """
    Plot lmfit time-domain result.
    INPUTS:
        x_data: x-axis (wavenumber) for fit_td
        Fit: lmfit object result from model.fit_td()

    """
    y_datai = Fit.data
    fit_datai = Fit.best_fit
    weight = Fit.weights
    # plot frequency-domain fit_td
    data_lessbl = np.real(np.fft.rfft(y_datai - (1 - weight) * (y_datai - fit_datai)))
    model = np.real(np.fft.rfft(fit_datai))
    # plot with residual_td
    fig, axs = plt.subplots(2, 1, sharex="col")
    axs[0].plot(x_data, data_lessbl, x_data, model)
    axs[1].plot(x_data, data_lessbl - model)
    axs[0].set_ylabel("Absorbance")
    # axs[0].legend(['data','fit_td'])
    axs[1].set_ylabel("Residual")
    axs[1].set_xlabel("Wavenumber ($cm^{-1}$)")
    if tx_title:
        t_fit = Fit.best_values["temperature"]
        x_fit = Fit.best_values["molefraction"]
        axs[0].set_title(
            "Combustor fit_td T = "
            + f"{t_fit:.0f}"
            + "K, "
            + f"{100 * x_fit:.1f}"
            + "% H2O"
        )
    # and time-domain fit_td
    plt.figure()
    plt.plot(fit_datai)
    plt.plot(y_datai - fit_datai)
    plt.plot(weight)
    #    plt.legend(['model','residual_td','weighting function'])

    return data_lessbl


def trim_bandwidth(x_wvn_full, trans_full, band_fit):
    """This function converts the intersting portion of the spectrum to the
    time domain. wrapper for functions in pldspectrapy.td_support.

    Parameters
    ----------
    x_wvn_full : ndarray
        _description_
    trans_full : ndarray
        _description_
    band_fit : _type_
        _description_

    Returns
    -------
    x_wvn_trimmed : ndarray
        trimmed frequency axis
    trans_trimmed : ndarray
        trimmed transmission spectrum
    """

    start_pnt, stop_pnt = bandwidth_select_td(
        x_array=x_wvn_full, band_fit=band_fit, max_prime_factor=1000
    )

    x_wvn_trimmed, trans_trimmed = bandwidth_trim(
        x_wvn_full=x_wvn_full,
        trans_full=trans_full,
        start_pnt=start_pnt,
        stop_pnt=stop_pnt,
        band_fit=band_fit,
    )

    return x_wvn_trimmed, trans_trimmed


def create_spectrum(daq_file, config_dict, averaging_indices=None):
    """Gets data from a daq_file object and returns the frequency axis,
    matrix of  transmission spectra, and averaging info for transmission matrix

    Parameters
    ----------
    daq_file : igtools.DAQFilesVC707
        DAQ file object
    config_dict : dictionary
        dictionary containing all the values parsed from the config file
        entry examples:
            "lock_freq": 32e6,
            "band_fit": [6007, 6155]
    averaging_indices : list, optional
        list of indices to average over. The default is None.

    Returns
    -------
    x_wvn_full : numpy array
        frequency axis
    transmission : numpy array
        transmission spectrum
    """

    if daq_file.x_wvn is None:
        daq_file.prep_for_processing(config_dict)
    x_wvn_full = daq_file.x_wvn

    if averaging_indices is None:
        averaging_indices = daq_file.avg_pc_data_info["indices"][0]

    pc_ig = avg_igs_by_idx(daq_file.data, averaging_indices)

    # TODO: should we be using rfft here to avoid the length mismatch?
    #  How does this play with the checking side of the nyquist window?
    transmission = np.abs(np.fft.fft(pc_ig))

    return x_wvn_full, transmission


def initialize_hapi_db(hapi_path):
    """Intializes the HAPI database according to specified path

    Parameters
    ----------
    hapi_path : str
        path to linelists directory

    Returns
    -------
    None
    """
    # Make sure either at least one .data or a .par file is present in the linelists directory
    if not os.path.exists(hapi_path):
        raise FileNotFoundError("No linelists directory found")
    # check for .data or .par files
    if not any(
        [
            file.endswith(".data") or file.endswith(".par")
            for file in os.listdir(hapi_path)
        ]
    ):
        raise FileNotFoundError("No .data or .par files found in linelists directory")

    pldhapi.db_begin(hapi_path)
    hapi2.hapi.db_begin(hapi_path)


def fit_data(
    x_wvn,
    transmission,
    config_variables,
    plot_intermediate_steps=False,
    verbose=False,
):
    """
    Fits spectrum to a model using the lmfit package and config file parameters

    Parameters
    ----------
    x_wvn : numpy array
        wavenumber array
    transmission : numpy array
        transmission array
    config_variables : dictionary
        contains all the values parsed from the config file
    debug_mode : boolean
        if True, skips the fitting and just returns the model evaluated at the input parameters
    plot_intermediate_steps : boolean
        if True, plots the intermediate steps of the fitting process. Useful for learning

    Returns
    -------
    Fit: lmfit.Model object
        fit object containing the model and parameters. Can be used to extract fit parameters and uncertainties

    """

    # Initialize progress bar
    pbar = tqdm(total=None, desc="Fitting progress")

    def callback(params, iter, resid, *args, **kwargs):
        """Callback function for updating tqdm progress bar"""
        pbar.update(1)

    x_wvn_trimmed, transmission_trimmed = pld.trim_bandwidth(
        x_wvn, transmission, config_variables["fitting"]["band_fit"]
    )
    y_td = pld.calc_cepstrum(transmission_trimmed)

    weight_array = pld.weight_func(
        len(x_wvn_trimmed),
        config_variables["fitting"]["baseline"],
        config_variables["fitting"]["etalons"],
    )

    if config_variables["fitting"]["simulation_backend"] == "gaas":
        config_variables = add_gaas_abs_db_to_config_dict(
            config_variables, x_wvn_trimmed
        )
    elif "scale" in config_variables["fitting"]["simulation_backend"]:
        config_variables = add_scale_model_to_config_dict(
            config_variables, x_wvn_trimmed
        )
    model, parameters = model_creation.setup_models(config_variables)

    # model.post_fit = post_process_fit
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
        # TODO: finish this function
        generate_tutorial_fit_plots(x_wvn, transmission, config_variables, Fit)

    return Fit


def add_gaas_abs_db_to_config_dict(config_variables, x_wvn_trimmed):
    # NOTE: this loop already happens in model_creation.create_composite_model(),
    # but we have to do here to get the absDB_dict into the config_variables

    fit_dict = config_variables["model_setup"]
    molecules = fit_dict.keys()
    # Remove "global" from the list of molecules. These are not molecules,
    # they are global parameters
    molecules = [molecule for molecule in molecules if molecule not in exclude_names]
    absDB_dict = {}
    for index, molecule in enumerate(molecules):
        # Create absDBF
        absDB = gs.gen_abs_db(
            fit_dict[molecule]["db_name"],
            fit_dict[molecule]["iso"],
            x_wvn_trimmed[0],
            x_wvn_trimmed[len(x_wvn_trimmed) - 1],
            config_variables["input"]["linelist_path"],
            0,  # line strength cutoff threshold
            loadFromHITRAN=False,
        )
        # Add absDB to absDB_dict
        absDB_dict[index] = absDB
        # Create pointer to absDB_dict key
        fit_dict[molecule]["gaas_key"] = int(index)
        # Add absDB_dict to config_variables['model_setup']
        fit_dict[molecule]["gaas_db"] = absDB_dict
    # Add update config_variables with modified fit_dict
    config_variables["model_setup"] = fit_dict

    return config_variables


def add_scale_model_to_config_dict(config_variables, x_wvn_input):
    """
    This function adds the data needed to run the scale-model fitting approach to the configuration dictionray
    Parameters
    ----------
    config_variables: dictionary
        Dictionary containing information on data processing/handling parameters.
    x_wvn_input: numpy arary
        Wavenumber axis associated with data being processed

    Returns
    -------
    Updated config_variables: inserts scale-model parameters into the fit information
    for each molecule in the config
    """
    fit_dict = config_variables["model_setup"]
    molecules = fit_dict.keys()
    molecules = [molecule for molecule in molecules if molecule not in exclude_names]
    scale_models = {}
    for index, molecule in enumerate(molecules):
        # Check if scalefactor model is already present and if it should be updated
        # If it is not present, or if it is present but should be updated, create a new scale model

        scalefactor_check = fit_dict.get(molecule, {}).get("update_model", True)
        if not scalefactor_check:
            return config_variables

        # Create scale model
        wvn, scale_model = pld.create_simulation_from_fit_config(
            config_variables, molecule
        )
        scale_models[index] = np.fft.irfft(np.interp(x_wvn_input, wvn, scale_model))
        # Add scalefactor model parameters
        fit_dict[molecule]["scalefactor"] = {
            "value": 1,
            "max": 1e4,
            "min": 1e-4,
            "vary": True,
        }
        # Create a key to identify the correct scale model
        fit_dict[molecule]["scale_key"] = int(index)
        # Add scale models to config_variables['model_setup']
        fit_dict[molecule]["scale_models"] = scale_models
        # Create a flag to indicate if this model will be updated and default to
        # 'False' and if needs to be 'True' should be set elsewhere
        fit_dict[molecule]["update_model"] = False
    # Add update config_variables with modified fit_dict
    config_variables["model_setup"] = fit_dict

    return config_variables


def compute_data_minus_baseline(data, weight, fit):
    """
    Compute the data minus the baseline using the weight and fit. This is usefull for plotting
    fit results.  This calcuation is done in the cepstrum and then converted back to the
    frequency domain.

    Parameters
    ----------
    data: array
        data array
    weight: array
        weight array - comes from weight_func
    fit: array
        fit array

    Returns
    -------
    data_less_bl: array
        data minus baseline array
    """

    data_less_bl = data - (1 - weight) * (data - fit)

    return data_less_bl


def get_cepstrum_residual(fit_result):
    """
    Extracts the residual array (data_minus_baseline - fit)
    from the fit object in the cepstrum domain.

    Parameters
    ----------
    fit_result: lmfit.Model object
        Fit object containing the model and parameters.

    Returns
    -------
    cepstrum_residual: array
        residual in the cepstrum domain
    """

    data_less_bl_time_domain = compute_data_minus_baseline(
        data=fit_result.data, weight=fit_result.weights, fit=fit_result.best_fit
    )

    cepstrum_residual = data_less_bl_time_domain - fit_result.best_fit

    return cepstrum_residual


def compute_cepstrum_residual_sdev(
    fit_result, reference_point=None, reference_width=1000
):
    """
    Compute the standard deviation of the residual array in the cepstrum domain.

    Parameters
    ----------
    fit_result: lmfit.Model object
        Fit object containing the model and parameters.

    reference_point: int, optional
        index point for the standard deviation calculation. The default is None.
        If none, will calculate in the middle of the array.

    reference_width: int, optional
        width of the calculation region. The default is 100.

    Returns
    -------
    cepstrum_residual_std: float
        standard deviation of the residual array over the reference range
    """

    residual_array = get_cepstrum_residual(fit_result)

    if reference_point is None:
        reference_point = int(len(residual_array) / 2)

    half_width = int(reference_width / 2)
    cepstrum_residual_std = np.std(
        residual_array[reference_point - half_width : reference_point + half_width]
    )

    return cepstrum_residual_std


def compute_absorbance_residual_sdev(
    fit_result, reference_wavenumber=6050.5, reference_width_wavenumber=0.5
):
    """
    Compute the standard deviation of the residual array in the absorbance domain.
    This will be of an area with no feature to get a sense of the noise level.

    Parameters
    ----------
    fit_result: lmfit.Model object
        Fit object containing the model and parameters.

    reference_wavenumber: float, optional
        index point for the standard deviation calculation. The default is None.
        If none, will calculate in the middle of the array.

    reference_width_wavenumber: float, optional
        width of the calculation region. The default is 100.

    Returns
    -------
    absorbance_residual_std: float
        standard deviation of the residual array over the reference range
    """

    cepstrum_residual = get_cepstrum_residual(fit_result)

    # Convert the residual back to the frequency domain
    residual = np.fft.rfft(cepstrum_residual)

    # Find the reference points
    small_bandwidth = [
        reference_wavenumber - reference_width_wavenumber / 2,
        reference_wavenumber + reference_width_wavenumber / 2,
    ]

    x_wavelength_axis = fit_result.userkws["xx"]
    _, small_residual = trim_bandwidth(
        x_wavelength_axis,
        residual,
        small_bandwidth,
    )

    return np.std(small_residual)
