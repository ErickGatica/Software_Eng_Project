import os
import warnings
import json5
import pandas as pd
import numpy as np

from pldspectrapy import (
    is_float,
    get_hitran_molecule_id,
    compute_cepstrum_residual_sdev,
    compute_absorbance_residual_sdev,
    # convert_time_params,
)

import pldspectrapy as pld


def parse_config_file(file):
    """
    This function parses a config file and returns its data as a dictionary to be accessed by other functions
    in the Spectral Fit program. It contains parameters that determine how the spectral fitting will be run.

    Parameters
    ----------
    file : string
        path / name of file to be parsed

    Returns
    -------
    InputData : dictionary
        dictionary containing all of the values parsed from the config file

    TODO:
    - handle arrays?
    """
    ConfigData = {"config_filename": file}

    current_section_name = None

    duplicate_fields = []

    with open(file, "r", encoding="utf-8-sig") as textfile:
        for line in textfile:
            line = line.strip()

            # line_number += 1
            # handle section breaks and empty lines
            if line.startswith("#") or len(line) == 0:
                continue

            # handle inline comments
            line = line.split("#")[0]

            # split line into field and value
            split_line = line.split(":")
            # replace spaces, make lowercase
            field_name = split_line[0].replace(" ", "").lower()

            field_value = split_line[1]
            field_value = field_value.lower().strip()

            # create new key in dictionary for each section
            if field_name == "section":
                current_section_name = field_value.replace(" ", "").lower()

                # check for duplicate sections
                if current_section_name in ConfigData:
                    duplicate_fields.append(current_section_name)

                ConfigData.update({current_section_name: {}})
                continue

            # check for duplicate fields
            if field_name in ConfigData[current_section_name]:
                duplicate_fields.append(field_name)

            # save values as float if they're numbers, as strings if not
            if is_float(field_value):
                ConfigData[current_section_name].update(
                    {field_name: float(field_value)}
                )
            else:
                ConfigData[current_section_name].update({field_name: field_value})

            # raise error if duplicate fields are found
        if duplicate_fields:
            for field in duplicate_fields:
                print('ERROR: multiple instances of "{}" found.'.format(field))
            raise ValueError("Mutiple values found for same field name")

    return ConfigData


def generate_output_dictionary_fitresults(fit_results, config_dict):
    """
    This function parses the Model results object and return a dictionary of values
    that we will want to either have access to, save out, or both

    Parameters
    ----------
    config_dict: dictionary
        Configuration dictionary for fitting
    fit_results: Model result object
        This is the output object from running Model.fit()

    Returns
    -------
    output_dict: dictionary
        Dictionary containing all of the Model fit parameters, their standard
        deviation values, and whether or not they were floated
    """

    output_dict = {}

    # db_name_list = config_dict["model_setup"]["db_names"]["value"]
    # The param list includes the database name even though we passed it as a string to the
    # fitting function.

    param_list = list(fit_results.best_values.keys())

    # TODO Handle the GAAS specific parameters better
    for param in param_list:
        if ("absDB" in param) or ("gaas" in param):
            continue
        # Add the value of the parameter to the output dictionary
        output_dict[param] = fit_results.best_values[param]

        if is_float(output_dict[param]):
            # If the fit matches the data exactly, there is no uvars attribute.
            # This causes a problem if you are fitting synthetic data as
            # the saving will fail.

            output_dict[param + "_vary"] = fit_results.init_params[param].vary
            try:
                output_dict[param + "_sdev"] = fit_results.uvars[param].std_dev
            except AttributeError:
                output_dict[param + "_sdev"] = 0

        # if the parameter was specificed with an expression (i.e., a global value)
        # add the expression to the output dictionary.
        try:
            expr = fit_results.init_params[param].expr
            if expr is not None:
                output_dict[param + "_expr"] = expr
        except KeyError:
            pass

    return output_dict


def generate_output_dictionary_fitparams(config_dict):
    """
    This function converts the fitting parameters in the configuration dictionary to
    a separate dictionary for writing to the output file

    Parameters
    ----------
    config_dict: dictionary
        Configuration dictionary for fitting

    Returns
    -------
    output_dict: dictionary
        Dictionary containing only the fitting parameters
    """
    output_dict = dict(config_dict["fitting"])
    band_fit = []
    band_fit.append(list(config_dict["fitting"]["band_fit"]))
    output_dict["band_fit"] = band_fit
    etalons = []
    etalons.append(list(config_dict["fitting"]["etalons"]))
    output_dict["etalons"] = etalons
    return output_dict


def generate_output_df(
    config_dict, results_dict, user_output_dict=None, existing_df=None
):
    """
    This function combines the dictionaries with the fitting parameters and the fit
    results into a single pandas DataFrame
    Parameters
    ----------
    user_output_dict
    config_dict: dictionary
        Configuration dictionary used for fitting
    results_dict: dictionary
        Dictionary with the lmfit Model results
    user_output_dict: dictionary
        User configured dictionary of variables to include in the output file
    existing_df: pandas DataFrame
        If appending to an existing dataframe, pass it here

    Returns
    -------
    output_df: pandas DataFrame
        DataFrame with combined information from the input dictionaries
    """

    save_flag = config_dict["input"]["save_results"]
    if save_flag:
        save_file = os.path.join(
            config_dict["input"]["results_path"], config_dict["input"]["results_name"]
        )
        # Get the directory from the save_file path.
        save_dir = os.path.dirname(save_file)

        # Check if the directory exists, if not, create it. Return error if fails.
        if not os.path.isdir(save_dir):
            try:
                warnings.warn(
                    f"Inputted results path {save_dir} does not currently exist - now creating this path..."
                )
                os.makedirs(save_dir)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to create the directory {save_dir}: {str(e)}"
                )

    else:
        save_file = None

    fitting_df = pd.DataFrame(generate_output_dictionary_fitparams(config_dict))
    results_df = pd.DataFrame(results_dict, index=[0])
    filename_df = pd.DataFrame(
        pd.Series(config_dict["input"]["filename"], name="filename", index=[0])
    )

    if user_output_dict is not None:
        user_output_df = pd.DataFrame(user_output_dict, index=[0])
        output_df = pd.concat(
            [filename_df, fitting_df, results_df, user_output_df], axis=1
        )
    else:
        output_df = pd.concat([filename_df, fitting_df, results_df], axis=1)

    try:
        format_dict = config_dict["output"]
        if not format_dict["apply"]["value"]:
            format_dict = None
    except:
        format_dict = None

    if format_dict is not None:
        if (len(format_dict) - 1) != len(output_df.columns):
            # TODO This is only marginally helpful - should explicitely state errors
            raise Exception(
                "Format dictionary and results dataframe do not contain "
                "the same number of elements"
            )
        for param, format in format_dict.items():
            if param == "apply":
                continue
            if format["save"]:
                # print("Param = ", param, "; format = ", format["format"])
                output_df[param] = output_df[param].apply(format["format"].format)
            else:
                output_df.drop(param, axis=1, inplace=True)

    if save_file is not None:
        save_output_df(output_df, save_file)
    # elif save_file is not None and format_dict is None:
    #     save_output_df(output_df, save_file)

    if existing_df is not None:
        output_df = pd.concat([existing_df, output_df], axis=0, ignore_index=True)

    return output_df


def print_formatted_results(config_dict, results_df, reload_data=True):
    """
    This function is used to check/verify the configured formatting for the
    outputting before running the save function

    Parameters
    ----------
    config_dict: dictionary
        Fitting configuration dictionary
    results_df: pandas DataFrame
        Processing results stored in a DataFrame
    reload_data: bool
        Boolean indicating whether or not data should be reloaded from the test
        output file; default is True since the DataFrame needs to be regenerated
        in the event that certain columns are removed

    Returns
    -------
        Prints the formatted results DataFrame (including a view of columns that are
        specified as not being included in the output)
    """

    if reload_data:
        results_df = pd.read_csv(
            os.path.join(
                config_dict["input"]["results_path"],
                config_dict["input"]["results_name"],
            ),
            sep="\t",
        )

    try:
        format_dict = config_dict["output"]
    except:
        raise Exception("No formatting present in configuration dictionary")

    output_df = results_df.copy()

    for param, format in format_dict.items():
        if param == "apply":
            continue
        if format["save"]:
            # print("Param = ", param, "; format = ", format["format"])
            output_df[param] = output_df[param].apply(format["format"].format)
        else:
            output_df.drop(param, axis=1, inplace=True)

    try:
        display(output_df)  # Works with Jupyter notebook
    except:
        print(output_df.to_markdown())

    return output_df


def save_output_df(output_df, save_file):
    """
    This function saves the output data frame

    Parameters
    ----------
    output_df: pandas DataFrame
        This is the dataframe created by generate_output_df() which contains the
        fitting params and fitting results from a data processing run
    save_file: str
        Full file path to the output file for saving

    Returns
    -------
    No returns
    """

    if os.path.isfile(save_file):
        if check_file_header(output_df, save_file):
            output_df.to_csv(save_file, mode="a", header=False, index=False, sep="\t")
        else:
            raise Exception(
                "Current output file contains a different number of "
                "columns than the output dataframe"
            )
    else:
        output_df.to_csv(save_file, mode="w", index=False, sep="\t")


def generate_output_and_save(fit_results, config_dict, user_output_dict=None):
    """
    This function handles the creating and saving of the results of the fit

    Parameters
    ----------
    config_dict: dictionary
        Configuration dictionary used for setting up the fitting
    fit_results: lmfit Model results object
        This is the output/results from running Model.fit()

    Returns
    -------
    If selected (in configuration) data is written to the specified file; also
    returns the structured dataframe with the results
    """

    results_dict = generate_output_dictionary_fitresults(fit_results, config_dict)
    output_df = generate_output_df(
        config_dict, results_dict, user_output_dict=user_output_dict
    )

    return output_df


def generate_output_configuration(config_dict, config_file=None):
    """
    This function reads an example results file and creates a template dictionary for
    controlling the output parameter formatting for future file saving

    If a configuration file is specified, the function will write the modified
    configuration dictionary to that file

    Parameters
    ----------
    config_dict: dictionary
        The configuration dictionary used for setting fitting parameters
    config_file: str
        Full path to the configuration file (specify to overwrite the original
        configuration)

    Returns
    -------
    output_config_dict: dictionary
        Returns the input configuration dictionary with the added section that
        controls the output format
    """
    output_config_dict = config_dict.copy()

    results_df = read_results(config_dict)

    results_config_dict = {"apply": {"value": True}}

    # TODO Update default formatting values to something more useful (i.e.,
    #  read parameter and optimize the formatting)
    for param in list(results_df.columns):
        results_config_dict[param] = {"save": True, "format": "{:}"}

    output_config_dict["output"] = results_config_dict

    if config_file is not None:
        with open(config_file, "w") as json_file:
            json5.dump(output_config_dict, json_file, indent=4)

        reformat_json5_config(config_file)

    return output_config_dict


def generate_format_dict(config_dict):
    """
    This function reads the 'output' section of the configuration dictionary and
    creates a smaller dictionary that is fed into DataFrame.to_csv() as a kwarg for
    controlling the formatting of the individual columns

    Parameters
    ----------
    config_dict: dictionary
        Fitting parameter configuration dictionary

    Returns
    -------
    format_dict:
        Specific dictionary for passing to DataFrame.to_csv()
    """

    try:
        format_param_dict = config_dict["output"]
    except:
        print("No configuration for output formatting present in dictionary")
        return None

    format_dict = {}

    for param in format_param_dict.items():
        format_dict[param[0]] = param[1]["format"]

    return format_dict


def load_config_json(config_path):
    """
    Loads the config file and returns the config variables in a dictionary

    Parameters
    ----------
    config_path : string
        path / name of file to be parsed
    test_config_path : string
        path / name of file to be parsed for testing purposes

    Returns
    -------
    config_variables : dictionary
        dictionary containing all the values parsed from the config file
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Config file not found")

    with open(config_path, "r") as f:
        config_data = json5.load(f)

    return config_data


def create_config_json(config_path):
    """
    This function creates a template json5 config file for data processing

    Parameters
    ----------
    config_path: str
        Full file path including file name for the new config file (json5 format)

    Returns:
    ----------
    Creates the specified file - no functional returns
    """
    # TODO: PyCharm is pretty mad that these are not strings. I think we need to
    #  change them to strings so it isn't trying to write them as variables.

    template_data = {
        input: {
            data_path: "C:/data",
            filename: "20240205170949",
            process_raw: False,
            linelist_path: "C:/linelists",
            fit_plots: True,
            save_plot: True,
            plot_path: "C:/plots",
            plot_name: "fit_plot.png",
            save_results: True,
            results_path: "C:/output",
            results_name: "fit_output.csv",
        },
        fitting: {
            band_fit: "list of indices",
            baseline: "baseline index",
            etalons: "list of list of indices",
            lock_frequency: "lock frequency in Hz (typically 32e6)",
            simulation_backend: "hapi",
        },
        model_setup: {
            "db_names": {"list of db file names"},
            "molecule name": {
                pressure: {
                    value: "pressure in atmospheres",
                    vary: "True or " "False",
                    min: 0.01,
                    max: 1,
                },
                temperature: {
                    value: "temperature in Kelvin",
                    vary: "True or " "False",
                    min: 200,
                    max: 5000,
                },
                molefraction: {
                    value: "relative mole fraction",
                    vary: "True or " "False",
                    min: 0,
                    max: 1,
                },
                pathlength: {"value": "pathlength in cm", "vary": False},
                iso: {"value": "molecular isotope number", "vary": False},
                mol_id: {"value": "HITRAN molecule ID", "vary": False},
                shift: {"value": 0, "vary": "True or False", "min": -0.2, "max": 0.2},
                db_name: "Name of database file",
            },
        },
    }
    with open(config_path, "w") as json_file:
        json5.dump(template_data, json_file, indent=4)


def set_molecule_config_params(input_dict, molecule_name, molefraction={}):
    """
    This function sets some of the molecular parameters in the configuration
    dictionary used for setting up data processing. Currently searches a HITRAN list
    of molecules to pull the HITRAN molecular ID, sets a default value of 1 for the
    isotope number, and sets a default molefraction if the molecule is H2O, CO2, or CH4

    Parameters
    ----------
    input_dict: dictionary
        The configuration dictionary generated for data processing

    molecule_name: str
        Name of the molecule for which parameters are being set

    molefraction: dictionary
        Dictionary for the molefraction parameters of the data processing configuration

    Returns
    -------
    input_dict: with updated entries for the named molecule

    """
    # Search for molecule information in HITRAN list
    try:
        print("Searching HITRAN for ", molecule_name)
        molec_id = get_hitran_molecule_id(molecule_name)
    except:
        raise Exception("Error finding specified molecule in HITRAN list")

    # Set the molecule ID from above and the isotope number to default of 1
    input_dict["model_setup"][molecule_name]["mol_id"]["value"] = int(molec_id)
    input_dict["model_setup"][molecule_name]["iso"]["value"] = int(1)

    try:
        molefraction_value = molefraction["value"]
    except:
        if molecule_name.lower() == "h2o":
            molefraction_value = 0.01
        elif molecule_name.lower() == "co2":
            molefraction_value = 450e-6
        elif molecule_name.lower() == "ch4":
            molefraction_value = 2e-6

    try:
        molefraction_min = molefraction["min"]
    except:
        molefraction_min = molefraction_value * 0.5

    try:
        molefraction_max = molefraction["max"]
    except:
        molefraction_max = molefraction_value * 1.5

    try:
        molefraction_vary = molefraction["vary"]
    except:
        molefraction_vary = True

    input_dict["model_setup"][molecule_name]["molefraction"][
        "value"
    ] = molefraction_value
    input_dict["model_setup"][molecule_name]["molefraction"]["vary"] = molefraction_vary
    input_dict["model_setup"][molecule_name]["molefraction"]["min"] = molefraction_min
    input_dict["model_setup"][molecule_name]["molefraction"]["max"] = molefraction_max

    if molecule_name.lower() == "ch4":
        input_dict["model_setup"][molecule_name]["db_name"]["value"] = (
            molecule_name.upper() + "_HIT08"
        )
    else:
        input_dict["model_setup"][molecule_name]["db_name"][
            "value"
        ] = molecule_name.upper()

    return input_dict


def update_config_dictionary(input_dict, param):
    """
    This function modifies the data processing configuration dictionary based on
    keywords and using some default values. Currently only supports changing the
    temperature, pressure, and shift fields within the 'model_setup' portion of the
    dictionary.

    Parameters
    ----------
    input_dict: dictionary
        Dictionary containing the configuration parameters for data processing

    param: str
        Name of the parameter being modified

    Returns
    -------
    input_dict
        User input inserted where specified and default values used for other
        required parameters
    """
    # TODO Need to create a way to handle passing expressions instead of actual fit
    #  parameters. Could probably just be a check that jumps to a different portion
    #  of code that adds the expression rather than creating all the associated params

    # TODO: Move these to a CONSTANTS file or at the top of the file.
    if param == "temperature":
        default_value = 300
        default_min = 200
        default_max = 5000
    elif param == "pressure":
        default_value = 1
        default_min = 0.01
        default_max = 1.1
    elif param == "shift":
        default_value = 0
        default_min = -0.2
        default_max = 0.2
    default_vary = False

    try:
        value = input_dict["value"]
    except:
        value = default_value

    try:
        vary = input_dict["vary"]
    except:
        vary = default_vary

    try:
        min_ = input_dict["min"]
    except:
        min_ = default_min

    try:
        max_ = input_dict["max"]
    except:
        max_ = default_max

    input_dict["value"] = value
    input_dict["vary"] = vary
    input_dict["min"] = min_
    input_dict["max"] = max_

    return input_dict


def default_molecule():
    """
    This function creates an empty dictionary with the structure for adding molecules
    to the data processing configuration dictionary

    Returns
    -------
    default_molec_dict
    """
    default_molec_dict = {
        "pressure": {
            "value": "pressure in atmospheres",
            "vary": "True or " "False",
            "min": 0.01,
            "max": 1,
        },
        "temperature": {
            "value": "temperature in Kelvin",
            "vary": "True or " "False",
            "min": 200,
            "max": 5000,
        },
        "molefraction": {
            "value": "relative mole fraction",
            "vary": "True or " "False",
            "min": 0,
            "max": 1,
        },
        "iso": {"value": "molecular isotope number", "vary": False},
        "mol_id": {"value": "HITRAN molecule ID", "vary": False},
        "shift": {"value": 0, "vary": "True or False", "min": -0.2, "max": 0.2},
        "db_name": {"value": "Name of database file"},
    }
    return default_molec_dict


def addmolecule_config_json(config_file, molecule_name, **kwargs):
    """
    This function inserts new molecules into the data processing configuration
    dictionary/file based on user inputs and default values

    Parameters
    ----------
    config_file: str
        Full file path to the configuration file being modified

    molecule_name: str
        Name of the molecule for which parameters are being modified (needs to match
        HITRAN molecule names - not necessarily case sensitive)
    kwargs: dictionaries
        temperature & pressure
        Format example:
        temperature={'value':300, 'vary':False, 'min':270, 'max':350}
        Note that not all parameters are required, but ones not specified will be set to default values

    Returns
    -------
    No returns but overwrites specified configuration file with updated version
    """
    config_data = load_config_json(config_file)

    if list(config_data["model_setup"].keys()) == ["molecule name"]:
        config_data["model_setup"][molecule_name] = config_data["model_setup"][
            "molecule name"
        ]
        del config_data["model_setup"]["molecule name"]
    else:
        config_data["model_setup"][molecule_name] = default_molecule()

    try:
        molefraction = kwargs["molefraction"]
        config_data = set_molecule_config_params(
            config_data, molecule_name, molefraction=molefraction
        )
    except:
        config_data = set_molecule_config_params(config_data, molecule_name)

    try:
        pressure_dict = kwargs["pressure"]
        if "expr" not in list(pressure_dict.keys()):
            pressure_dict = update_config_dictionary(pressure_dict, "pressure")
    except:
        pressure_dict = {"value": 1, "vary": False, "min": 0.01, "max": 1}

    try:
        temperature_dict = kwargs["temperature"]
        if "expr" not in list(temperature_dict.keys()):
            temperature_dict = update_config_dictionary(temperature_dict, "temperature")
    except:
        temperature_dict = {"value": 300, "vary": False, "min": 200, "max": 5000}

    try:
        shift_dict = kwargs["shift"]
        if "expr" not in list(shift_dict.keys()):
            shift_dict = update_config_dictionary(shift_dict, "shift")
    except:
        shift_dict = {"value": 0, "vary": False, "min": -0.2, "max": 0.2}

    try:
        pathlength_dict = kwargs["pathlength"]
        if "expr" not in list(pathlength_dict.keys()):
            pathlength_dict = update_config_dictionary(pathlength_dict, "pathlength")
    except:
        pathlength_dict = {"value": 0, "vary": False}

    try:
        db_name_dict = kwargs["db_name"]
        if "expr" not in list(db_name_dict.keys()):
            db_name_dict = update_config_dictionary(db_name_dict, "db_name")
    except:
        db_name_dict = {"value": 0, "vary": False}

    config_data["model_setup"][molecule_name]["pressure"] = pressure_dict
    config_data["model_setup"][molecule_name]["temperature"] = temperature_dict
    config_data["model_setup"][molecule_name]["shift"] = shift_dict
    config_data["model_setup"][molecule_name]["pathlength"] = pathlength_dict
    config_data["model_setup"][molecule_name]["db_name"] = db_name_dict

    with open(config_file, "w") as json_file:
        json5.dump(config_data, json_file, indent=4)


def check_file_header(output_df, save_file):
    """
    This function verifies that the same number of columns are present in the
    existing results file and the results dataframe that is prepared for appending to
    the results file
    Parameters
    ----------
    output_df: pandas DataFrame
        Formatted results dataframe
    save_file: str
        Full file path to the results file

    Returns
    -------
        Boolean indicating whether the number of columns matches or not
    """
    results_df = pd.read_csv(
        save_file,
        sep="\t",
    )
    if len(output_df.columns) == len(results_df.columns):
        return True
    else:
        return False


def read_results(config_dict):
    """
    This function reads a results file generated by the fitting process

    Parameters
    ----------
    config_dict: dictionary
        Fitting parameter configuration dictionary which also contains output file
        information

    Returns
    -------
    results_df: pandas DataFrame
        Results from data processing
    """

    try:
        results_df = pd.read_csv(
            os.path.join(
                config_dict["input"]["results_path"],
                config_dict["input"]["results_name"],
            ),
            sep="\t",
        )
    except:
        raise Exception(
            "Error loading configured results file - check configuration "
            "parameters and results file"
        )

    return results_df


def reformat_json5_config(input_file):
    """
    This function reformats the output from json5.dump into a more usable format
    Parameters
    ----------
    input_file: str
        Full file path to json5 configuration file to reformat

    Returns
    -------
    No return. Overwrites the input json5 file with the reformatted version
    """

    # TODO Read folder path from input_file and create dummy file in same directory
    output_file = r"examples\configs\test.json5"
    indent_level = 8

    before_output = True

    with open(input_file, "r") as infile:
        with open(output_file, "w") as outfile:
            cntr = 1
            for line in infile:
                current_indent_level = len(line) - len(line.lstrip())
                # print("Line ", cntr, " has an indent level of ", current_indent_level)
                if (before_output == True) and ("output:" in line):
                    before_output = False
                if (current_indent_level == indent_level) and ("]" not in line):
                    if ("," in line) or ("{" in line):
                        if before_output:
                            # print("Line ", cntr, ": conditional statement 1")
                            pass
                        else:
                            if "}," in line:
                                # print("Line ", cntr, ": conditional statement 2")
                                line = line.strip()
                                line += "\n"
                            else:
                                # print("Line ", cntr, ": conditional statement 3")
                                line = line.rstrip(" \r\n")
                    else:
                        # print("Line ", cntr, ": conditional statement 4")
                        line = line.rstrip(" \r\n")
                elif (current_indent_level == indent_level) and ("]" in line):
                    # print("Line ", cntr, ": conditional statement 5")
                    line = line.strip()
                    line += "\n"
                elif current_indent_level > indent_level:
                    if "{:" in line:
                        # print("Line ", cntr, ": conditional statement 6")
                        line = line.rstrip(" \r\n")
                        line = line.strip()
                    elif "{" in line:
                        # print("Line ", cntr, ": conditional statement 7")
                        line = line.rstrip(" \r\n")
                    elif "}," in line:
                        # print("Line ", cntr, ": conditional statement 8")
                        line = line.strip()
                        line += "\n"
                    else:
                        # print("Line ", cntr, ": conditional statement 9")
                        line = line.rstrip(" \r\n")
                        line = line.strip()
                elif (current_indent_level < indent_level) and ("{" in line):
                    # print("Line ", cntr, ": conditional statement 10 - no action taken")
                    pass

                outfile.write(line)
                cntr += 1
    with open(output_file, "r") as infile:
        with open(input_file, "w") as outfile:
            cntr = 0
            for line in infile:
                line = line.replace(",", ", ")
                line = line.rstrip(" \r\n")
                line += "\n"
                outfile.write(line)
                cntr += 1

    os.remove(output_file)


def validate_config(config_variables):
    # TODO: It may be nicer to move a lot of the error handling to this function
    #  rather than having it across in the individual functions. I'll start it with an example
    #  of how to handle the case where the config file is missing a required field and we
    #  can build from there.

    # If the user specifies that they want to save the plots but don't make them,
    # the plots won't actually be saved. Let's raise a warning if this is the case.
    if (
        config_variables["input"]["save_plot"]
        and not config_variables["input"]["fit_plots"]
    ):
        warnings.warn(
            "You have specified that you want to save the plots, but you have not "
            "requested that they be created. The plots will not be saved."
        )

    return None


def create_user_calculated_output_dictionary(fit_result, config_dict):
    """
    Create a dictionary of user-calculated output values that are not directly
    related to the fit results.

    Note that this function takes in the config dictionary,
    but it isn't used at this time.  This is a placeholder for future functionality.


    Parameters
    ----------
    fit_result: lmfit.Model result object
        This is the output object from running Model.fit()

    config_dict: dictionary
        Configuration dictionary for fitting

    Returns
    -------
    user_output_dict: dictionary
        Dictionary containing the user-calculated output values

    """

    time_domain_residaul_std = compute_cepstrum_residual_sdev(
        fit_result, reference_point=1500, reference_width=1000
    )

    absorbance_residual_std = compute_absorbance_residual_sdev(fit_result)

    function_evals = fit_result.nfev

    user_output_dict = {
        "td_res_sdev": time_domain_residaul_std,
        "abs_res_sdev": absorbance_residual_std,
        "function_evals": function_evals,
    }

    return user_output_dict


#
# def create_averaging_variables(
#     value,
#     daq_file,
#     config_variables,
#     method="number",
#     start=0,
#     end=0,
#     total=0,
# ):
#     """
#     This function creates the averaging_variables dictionary based on the input
#     parameters. It allows for number-based or time-based averaging, and then converts all
#     time-based parameters to index-based parameters.
#
#     Parameters
#     ----------
#     value: float
#         Base number controlling the averaging - if indexed-based, this is the number
#         of IGs to average; if time-based, this is the averaging time (in seconds)
#     daq_file: DaqFile object
#         Daq file to be processed
#     config_variables: dictionary
#         Dictionary of data processing parameters
#     method: string
#         String indicating time- or indexed-based averaging. Use 'number' for
#         indexed-based, and 'time' for time-based. Upon
#     start: int, optional
#         Start value for averaging (default is 0)
#     end: int, optional
#         End value for averaging (default is 0)
#     total: int, optional
#         Total averaging (default is 0)
#
#     Returns
#     -------
#     dict
#         Averaging variables dictionary with all parameters converted to index-based
#     """
#     if method not in ["number", "time"]:
#         raise ValueError("Averaging method must be 'time' or 'number'")
#
#     averaging_variables = {
#         # Variables associated with number-based averaging
#         "num": int(value) if method == "number" else float(value),
#         "start": int(start) if method == "number" else float(start),
#         "end": int(end) if method == "number" else float(end),
#         "total": int(total) if method == "number" else float(total),
#     }
#
#     # Immediately convert time-based parameters to index-based parameters.
#     # This makes all downstream processing much easier.
#     if method == "time":
#         averaging_variables = convert_time_params(averaging_variables, base_time)
#
#     averaging_variables = check_averaging_variables(
#         daq_file, config_variables, averaging_variables
#     )
#
#     return averaging_variables


def update_averaging_variables(
    averaging_variables, param_updates, method, daq_file, config_variables
):
    """
    This function modifies the contents of the averaging_variables dictionary and
    then checks the parameters for compatibility
    Parameters
    ----------
    averaging_variables: dictionary
        Dictionary containing parameters to control averaging
    param_updates: dictionary
        Dictionary of parameter: value pairs to update in averaging_variables
    method: str
        Flag for index-based ('number') or time-based ('time') value
    daq_file: DaqFile object
        DaqFile being processed
    config_variables: dictionary
        Dictionary of processing parameters

    Returns
    -------
    Modified averaging_variables dictionary
    """
    param_list = ["num", "start", "end", "total"]

    for param, value in param_updates.items():

        if param not in param_list:
            raise Exception(
                "Invalid param specified; must be 'start', 'end', or 'total'"
            )

        if method == "number":
            averaging_variables[param] = int(value)
        elif method == "time":
            averaging_variables[param] = int(value / daq_file.base_time)
        else:
            raise Exception("Invalid method specified; must be 'number' or 'time'")

    averaging_variables = check_averaging_variables(
        daq_file, config_variables, averaging_variables
    )

    return averaging_variables
