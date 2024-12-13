# -*- coding: utf-8 -*-
# pycodestyle: noqa

"""
Created February 26, 2024

@author: seco2718

This module contains functions specifically related to creating absorbance
simulations using the pldspectrapy repository as the backend.

Attempting to keep this closely related to the data processing functions

"""

import os
import warnings

import json5
import numpy as np
import pldspectrapy as pld
import hapi

from pldspectrapy.constants import SPEED_OF_LIGHT


param_list = [
    "pressure",
    "temperature",
    "molefraction",
    "pathlength",
    "isotope",
    "linelist_path",
    "database_name",
    "simulation_range",
    "simulation_resolution",
    "simulation_backend",
]


def save_dict_to_json(dictionary, config_path):
    """
    Saves a dictionary to a json5 file

    Parameters
    ----------
    dictionary: dictionary
        Dictionary to be saved to a json5 file
    config_path: str
        Full file path to save the dictionary

    Returns
    -------
    None
    """
    with open(config_path, "w") as json_file:
        json5.dump(dictionary, json_file, indent=4)


def create_simulation_dictionary(molecule_name="H2O"):
    """
    Creates a template dictionary for the simulation parameters.
    The default molecule added to the dictionary is "H2O".

    Parameters
    ----------
    molecule_name: str
        HITRAN name of molecule to add to simulation config

    Returns
    ----------
    template_data: dictionary
        Dictionary containing the default simulation parameters
    """
    template_data = {
        "input": {
            "plot": True,
            "save": True,
            "plot_path": "C:/plots",
            "plot_name": "simulation_plot.png",
            "save_path": "C:/output",
            "save_name": "simulation_output.csv",
        },
    }

    molecule_dict = default_molecule_simulation(molecule_name)
    template_data["model_setup"] = molecule_dict

    return template_data


def default_molecule_simulation(molecule_name="H2O"):
    """
    Creates an empty dictionary with the structure for adding molecules
    to the simulation configuration dictionary

    Parameters
    ----------
    molecule_name: str
        HITRAN name of molecule to add to simulation config

    Returns
    -------
    default_molec_dict: dictionary
        Dictionary containing the default simulation parameters for a specific
        molecule
    """
    molecule_id = pld.get_hitran_molecule_id(molecule_name)

    default_molecule_dict = {
        molecule_name: {
            "pressure": "pressure in atm",
            "temperature": "temperature in K",
            "pathlength": "pathlength in cm",
            "molefraction": "molefraction",
            "molecule_id": int(molecule_id),
            "isotope": "isotope #",
            "linelist_path": "C:/linelists",
            "database_name": "name of database file to use",
            "simulation_range": "range to simulate in cm^-1",
            "simulation_resolution": "spectral resolution of simulation in Hz",
            "simulation_backend": "hapi",
        }
    }

    return default_molecule_dict


def add_molecule_simulation(simulation_input, molecule_name, **kwargs):
    """
    Modifies the input simulation dictionary to add or modify the
    parameters for a specific molecule. The input dictionary can be either a full file
    path to configuration json5 file containing simulation parameters, or the simulation
    parameter dictionary created (and saved) by the create_simulation_config() function

    Parameters
    ----------
    simulation_input: dictionary
        Dictionary containing the simulation parameters
    molecule_name: str
        Name of the molecule for which parameters are being modified (needs to match
        HITRAN molecule names - not necessarily case sensitive)
    kwargs:
        temperature: float; in K
        pressure: float; in atm
        pathlength: float; in cm
        molefraction: float
        isotope: int or list of ints (isotope number to simulate - see HITRAN for
        molecular isotope numbers
            database_name: str; database file name to use for simulating
        simulation_range: 2 member list of floats; in cm^-1; wavenumber bounds for
            simulation
        simulation_resolution: float; in Hz; spectral resolution for simulation
        simulation_backend: str; simulation modes - accepted values = "hapi",
            "sd_hapi", "radis"

    Returns
    -------
    modified_simulation_input: dictionary
        Dictionary containing the modified simulation parameters
    """

    config_data = simulation_input

    if molecule_name not in list(config_data["model_setup"].keys()):
        config_data["model_setup"][molecule_name] = default_molecule_simulation(
            molecule_name=molecule_name
        )[molecule_name]

        warnings.warn(
            f"Molecule {molecule_name} not found in simulation dictionary. "
            f"Adding default parameters for {molecule_name}"
        )

    update_molecule_parameters(config_data, molecule_name, **kwargs)

    return config_data


def update_molecule_parameters(config_data, molecule_name, **kwargs):
    invalid_params = set(kwargs.keys()) - set(param_list)

    if len(invalid_params) > 0:
        raise Exception(
            f"Invalid parameters passed to add_molecule_simulation(): {invalid_params}"
        )

    config_data["model_setup"][molecule_name].update(kwargs)


def get_simulation_param(molecule_dict, param):
    """
    Returns the value of a specific parameter from the molecular dictionary

    Parameters
    ----------
    molecule_dict: dictionary
        Dictionary containing the simulation information for a particular molecule
    param: str
        Parameter from molecular dictionary to return

    Returns
    -------
    molecule_dict[param]: value of the parameter from the molecular dictionary

    """
    return molecule_dict[param]


def check_simulation_dictionary(simulation_dict):
    # TODO: Rewrite this function as a decerator to the generate_simulated_spectrum()
    #  and any other functinos that require a simulation dictionary.
    """

    Parameters
    ----------
    simulation_dict: dictionary
        Dictionary containing simulation parameters

    Returns
    -------
    Boolean indicating whether or not the input dictionary has the correct elements
    """

    if (list(simulation_dict.keys())[0] != "input") or (
        list(simulation_dict.keys())[1] != "model_setup"
    ):
        raise Exception(
            "Primary dictionary keys do not match. Expected 'input' and "
            "'model_setup'; current keys ",
            list(simulation_dict.keys()),
        )

    molecular_dict = simulation_dict["model_setup"]
    molecular_keys = [
        "pressure",
        "temperature",
        "pathlength",
        "molefraction",
        "molecule_id",
        "isotope",
        "linelist_path",
        "database_name",
        "simulation_range",
        "simulation_resolution",
        "simulation_backend",
    ]

    for molecule in list(molecular_dict.keys()):
        molec_keys = list(molecular_dict[molecule].keys())
        for param in molecular_keys:
            if param not in molec_keys:
                print(
                    "Molecular dictionary for "
                    + molecule
                    + " is missing the parameter "
                    + param
                )

    warnings.warn(
        "The simulation dictionary checking function is currently under construction"
    )
    return True


def generate_simulation_xaxis(molecule_dict):
    """
    Generates the wavenumber axis for the molecular simulation

    Parameters
    ----------
    molecule_dict: dictionary
        Dictionary containing the parameters for simulation. Relevent for this
        function are: 'simulation_range' and 'simulation_resolution'

    Returns
    -------
    wvn_arr: numpy array
        Wavenumber axis
    """

    # Convert spectral resolution in Hz to cm^-1
    wvn_resolution = (
        (SPEED_OF_LIGHT / molecule_dict["simulation_resolution"]) * 100
    ) ** -1
    # Calculate the number of points in the wavenumber axis based on the total range
    # and the resolution
    wvn_pnts = (
        np.round(
            (
                molecule_dict["simulation_range"][1]
                - molecule_dict["simulation_range"][0]
            )
            / wvn_resolution
        )
        + 1
    )

    # Calculate the wavenumber axis
    wvn_arr = (
        molecule_dict["simulation_range"][0] + np.arange(wvn_pnts) * wvn_resolution
    )

    return wvn_arr


def run_hapi_sim(molecule_dict, wvn):
    """

    Parameters
    ----------
    molecule_dict: dictionary
        Dictionary containing the simulation information for a particular molecule
    wvn: numpy array
        The wavenumber axis to be used for the molecular simulation

    Returns
    -------
        absorbance_coefficients: HAPI output from absorbtionCoefficient_Voigt()
        function; absorbance spectrum simulation
    """
    x_wvn, absorbance_coeficients = pld.absorptionCoefficient_Voigt(
        (
            (
                int(molecule_dict["molecule_id"]),
                int(molecule_dict["isotope"]),
                molecule_dict["molefraction"],
            ),
        ),
        molecule_dict["database_name"],
        HITRAN_units=False,
        OmegaGrid=wvn,
        Environment={"p": molecule_dict["pressure"], "T": molecule_dict["temperature"]},
        Diluent={
            "self": molecule_dict["molefraction"],
            "air": (1 - molecule_dict["molefraction"]),
        },
    )
    # assume natural abundance
    absorbance_coeficients *= hapi.abundance(
        int(molecule_dict["molecule_id"]), int(molecule_dict["isotope"])
    )

    # scale by pathlength
    absorbance_coeficients *= molecule_dict["pathlength"]

    return absorbance_coeficients


def run_sd_hapi_sim(molecule_dict, wvn):
    """

    Parameters
    ----------
    molecule_dict: dictionary
        Dictionary containing the simulation information for a particular molecule
    wvn: numpy array
        The wavenumber axis to be used for the molecular simulation

    Returns
    -------
        absorbance_coefficients: HAPI output from absorbtionCoefficient_SDVoigt()
        function; absorbance spectrum simulation
    """
    x_wvn, absorbance_coeficients = pld.absorptionCoefficient_SDVoigt(
        (
            (
                int(molecule_dict["molecule_id"]),
                int(molecule_dict["isotope"]),
                molecule_dict["molefraction"],
            ),
        ),
        molecule_dict["database_name"],
        HITRAN_units=False,
        OmegaGrid=wvn,
        Environment={"p": molecule_dict["pressure"], "T": molecule_dict["temperature"]},
        Diluent={
            "self": molecule_dict["molefraction"],
            "air": (1 - molecule_dict["molefraction"]),
        },
    )
    absorbance_coeficients *= hapi.abundance(
        int(molecule_dict["molecule_id"]), int(molecule_dict["isotope"])
    )  # assume natural abundance

    absorbance_coeficients *= molecule_dict["pathlength"]

    return absorbance_coeficients


def generate_simulated_spectrum(simulation_dict, sum_models=False, skip_db_begin=False):
    """

    Parameters
    ----------
    simulation_dict: dict
        simulation parameter dictionary created by create_simulation_dictionary() or
        loaded from a configuration file
    sum_models: bool
        Flag for summing the created models
    skip_db_begin: bool
        Flag to skip loading the linelist repository information (i.e., running
        db_begin())

    Returns
    -------
        output_dict: dictionary
            Dictionary containing: calculated absorbance models + wavenumber axes;
            summed model (if sum_models = True); and the original simulation
            configuration parameters
    """
    # Either load the config file or use the dictionary argument

    # TODO Add a function to check that the simulation_dictionary is structure correctly
    #  move this to a function decorator

    if not check_simulation_dictionary(simulation_dict):
        raise Exception(
            "Dictionary passed to 'generate_simulated_spectrum()' does "
            "not containing the correct data structures"
        )

    output_dict = {}

    # Iterate through molecules in simulation dictionary
    for idx, key in enumerate(simulation_dict["model_setup"].keys()):
        molecule_dict = simulation_dict["model_setup"][key]

        if idx == 0:
            linelist_path = molecule_dict["linelist_path"]
            if not skip_db_begin:
                pld.db_begin(linelist_path)
        else:
            if linelist_path != molecule_dict["linelist_path"]:
                linelist_path = molecule_dict["linelist_path"]
                pld.db_begin(linelist_path)

        # Generate the wavenumber axis
        x_wvn = generate_simulation_xaxis(molecule_dict)
        # Extract the simulation backend parameter
        sim_backend = get_simulation_param(molecule_dict, "simulation_backend")
        # Run absorbance simulation
        if sim_backend.lower() == "hapi":
            absorbance = run_hapi_sim(molecule_dict, x_wvn)
        elif sim_backend.lower() == "hapi_sd":
            absorbance = run_sd_hapi_sim(molecule_dict, x_wvn)
        elif sim_backend.lower() == "radis":
            raise NotImplementedError(
                "Radis backend not yet implemented for simulation"
            )

        output_dict["absorbance_" + key] = absorbance[:]

        if sum_models:
            if idx == 0:
                model_total = [0] * len(x_wvn)
            model_total += absorbance
            output_dict["x_wvn"] = x_wvn
        else:
            output_dict["x_wvn_" + key] = x_wvn

    if sum_models:
        output_dict["model_total"] = model_total

    output_dict["simulation_config"] = simulation_dict

    return output_dict


def create_simulation_from_fit_config(
    config_variables,
    molecule_name,
    resolution=100e6,
):
    """
    This function generates a model spectrum using information from a configuration
    dictionary rather than a simulation dictionary.
    Parameters
    ----------
    config_variables: dictionary
        Dictionary containing data processing/handling parameters
    molecule_name: str
        Name of molecule to model
    resolution:
        Spectral resolution for simulation in Hz

    Returns
    -------
    Wavenumber axis and absorbance spectrum generated from simulation
    """
    simulation_dict = pld.create_simulation_dictionary(molecule_name=molecule_name)
    simulation_dict["model_setup"][molecule_name]["molefraction"] = config_variables[
        "model_setup"
    ][molecule_name]["molefraction"]["value"]
    simulation_dict["model_setup"][molecule_name]["pressure"] = config_variables[
        "model_setup"
    ][molecule_name]["pressure"]["value"]

    simulation_dict["model_setup"][molecule_name]["isotope"] = int(
        config_variables["model_setup"][molecule_name]["iso"]["value"]
    )
    simulation_dict["model_setup"][molecule_name]["molecule_id"] = int(
        config_variables["model_setup"][molecule_name]["mol_id"]["value"]
    )
    simulation_dict["model_setup"][molecule_name]["temperature"] = config_variables[
        "model_setup"
    ][molecule_name]["temperature"]["value"]

    simulation_dict["model_setup"][molecule_name]["pathlength"] = config_variables[
        "model_setup"
    ][molecule_name]["pathlength"]["value"]

    simulation_dict["model_setup"][molecule_name]["linelist_path"] = config_variables[
        "input"
    ]["linelist_path"]
    simulation_dict["model_setup"][molecule_name]["database_name"] = config_variables[
        "model_setup"
    ]["ch4"]["db_name"]
    simulation_dict["model_setup"][molecule_name][
        "simulation_range"
    ] = config_variables["fitting"]["band_fit"]
    simulation_dict["model_setup"][molecule_name]["simulation_resolution"] = resolution
    simulation_dict["model_setup"][molecule_name]["simulation_backend"] = (
        config_variables["fitting"]["simulation_backend"]
    ).replace("scale_", "")

    simulation_output = pld.generate_simulated_spectrum(simulation_dict)

    return (
        simulation_output[f"x_wvn_{molecule_name}"],
        simulation_output[f"absorbance_{molecule_name}"],
    )
