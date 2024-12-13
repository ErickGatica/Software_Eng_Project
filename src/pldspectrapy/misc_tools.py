import os
import re
import pandas as pd

import numpy as np
from matplotlib import pyplot as plt

import pldspectrapy
from pldspectrapy.constants import MOLECULE_IDS


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def read_notes(daq_file):
    # TODO: Where should this live?  It's not really a misc tool
    m = re.search(
        r"beam '(?P<name>.*)'\(dist (?P<dist>\d*(?:\.\d*)?)\)",  # noqa: W605
        daq_file.notes,
        re.IGNORECASE,
    )
    if m is None:
        m = re.search(
            r"beam '(?P<name>.*)'",
            daq_file.notes,
            re.IGNORECASE,
        )
        return m["name"], float(0)
    else:
        return m["name"], float(m["dist"])


def process_fit_summary(Fit, save_file):
    """This function takes the lmfit Fit.summary() output and flattens
    it into a single level dictionary,
    which is then saved as a csv.

    Parameters
    ----------
    Fit : lmfit.Model object
        Fit object containing the model and parameters and results

    save_file : path
        path to save csv file

    Returns
    ----------
    processed_summary_dict : dictionary
        dictionary containing the flattened summary_dict
    """
    summary_dict = Fit.summary()

    output_dict = {}
    output_dict = {"rsquared": summary_dict["rsquared"]}

    for key in summary_dict["best_values"]:
        output_dict[key] = summary_dict["best_values"][key]

    output_df = pd.DataFrame(output_dict, index=[0])
    output_df.to_csv(save_file, index=False)

    return output_dict


def post_process_fit(result):
    from .td_support import compute_data_minus_baseline

    # TODO: fold this into the user-defined output functionality. This is a bit of a hack right now
    """
    Function to be called manually after fitting. Adds additional parameters to the Fit object.
    I'm not getting the built in post_fit function to work, so this is a workaround."

    Parameters
    ----------
    result : lmfit.Model object
        Fit object containing the model and parameters.

    Returns
    -------
    result_dict:
        Dictionary containing the fit results.
            Keys:
            "Fit": contains the Fit object
            "data_less_bl_time_domain": time domain date with the baseline removed
            "data_less_bl_freq_domain": frequency domain data with the baseline removed
            "model_frequency_domain": best fit model in the frequency domain
            "residual_frequency_domain": residual (data - model) in the frequency domain
            "x_wvn_trimmed": wavenumber axis for the fitted data
    """

    result_dict = {}

    data_less_bl_time_domain = compute_data_minus_baseline(
        data=result.data, weight=result.weights, fit=result.best_fit
    )

    data_less_bl_freq_domain = np.real(np.fft.rfft(data_less_bl_time_domain))

    result_dict["Fit"] = result
    result_dict["data_less_bl_time_domain"] = data_less_bl_time_domain
    result_dict["data_less_bl_freq_domain"] = data_less_bl_freq_domain
    result_dict["model_frequency_domain"] = np.real(np.fft.rfft(result.best_fit))
    result_dict["residual_frequency_domain"] = np.real(np.fft.rfft(result.residual))
    result_dict["x_wvn_trimmed"] = result.userkws["xx"]

    # TODO: this assumes the shift comes from water, which is not always the case.  Account for globals
    result_dict["x_wvn_out"] = result.userkws["xx"] + result.best_values["ch4_shift"]

    # Add dry mole fractions
    # result_dict["ch4_molefraction_dry"] = result.best_values["ch4_molefraction"] / (
    #     1 - result.best_values["h2o_molefraction"]
    # )
    # result_dict["co2_molefraction_dry"] = result.best_values["co2_molefraction"] / (
    #     1 - result.best_values["h2o_molefraction"]
    # )

    return result_dict


def check_str_for_chars(char_list, check_str):
    for char in char_list:
        if char in check_str:
            return True
    return False


def post_process_fit_old(result):
    """Function to be used by built-in lmfit post_fit function. Adds additional parameters to the Fit object
    Example usage:
        `Fit.post_fit = post_process_fit`
    Parameters
    ----------
    result : lmfit.Model object
        Fit object containing the model and parameters.

    Returns
    -------
    None
        This function is use in an inline fashion by the lmfit package. It modifies the Fit object in place.
    """

    data_less_bl_time_domain = result.data - (1 - result.weights) * (
        result.data - result.best_fit
    )

    data_less_bl_freq_domain = np.real(np.fft.rfft(data_less_bl_time_domain))

    result.params.add("data_less_bl_time_domain", value=data_less_bl_time_domain)
    result.params.add("data_less_bl_freq_domain", value=data_less_bl_freq_domain)
    result.params.add(
        "model_frequency_domain", value=np.real(np.fft.rfft(result.best_fit))
    )
    result.params.add(
        "residual_frequency_domain", value=np.real(np.fft.rfft(result.residual))
    )
    result.params.add("x_wvn_trimmed", value=result.userkws["xx"])
    result.params.add(
        "x_wvn_out", value=result.userkws["xx"] + result.best_values["h2oshift"]
    )

    # Add dry mole fractions
    result.params.add(
        "ch4_molefraction_dry",
        value=result.best_values["ch4molefraction"]
        / (1 - result.best_values["h2omolefraction"]),
    )
    result.params.add(
        "co2_molefraction_dry",
        value=result.best_values["co2molefraction"]
        / (1 - result.best_values["h2omolefraction"]),
    )


def get_hitran_molecule_id(molecule_name):
    """
    This function returns the HITRAN molecule ID for the named molecule
    Parameters
    ----------
    molecule_name: str
        Name of the molecule to search the HITRAN dictionary for - chemical formula
        needs to match the HITRAN entry for success

    Returns
    -------
    molecule_id: int
        The HITRAN molecule ID for the name molecule. Raises exception if molecule is not found

    """

    if molecule_name.upper() in MOLECULE_IDS:
        molecule_id = MOLECULE_IDS[molecule_name.upper()]
        return molecule_id
    else:
        molec_list = []
        [
            molec_list.append(idx + 1)
            for idx, molec in enumerate(MOLECULE_IDS)
            if molec in molecule_name
        ]
        if len(molec_list) > 0:
            return int(molec_list[0])
        else:
            raise Exception(f"{molecule_name} molecule does not exist in HITRAN")


def get_database_names(config_dict):
    """
    This function extracts the database names from the configuration dictionary.
    Eventually, this list gets passed to the SourceTables parameter
    in hapi.AbsorptionCoefficient_Voigt

    Parameters
    ----------
    config_dict: dictionary
        dictionary containing all the values parsed from the config file

    Returns
    -------
    db_list: list
        list of database names

    """

    db_list = []

    for idx, key in enumerate(config_dict["model_setup"].keys()):
        db_list.append(config_dict["model_setup"][key]["db_name"]["value"])

    return db_list


def check_for_data(data_path, file_name, find_raw=False):
    """
    This function checks for the existence of a daq file in the specified directory

    Parameters
    ----------
    data_path: str
        Full file path to the directory being checked
    file_name: str
        Name of the file (without extension) being checked for
    find_raw: bool
        If True, checks for the existence of a .raw data file
        in addition to a .cor in the specified directory

    Returns
    -------
    bool
        True if the file exists, False if it does not
    """

    if "." in file_name:
        raise ValueError("File name should not include file extension")

    file_types = [".log", ".cor"]
    if find_raw:
        # file_types.append(".raw")
        for idx in range(len(file_types)):
            if file_types[idx] == ".cor":
                file_types[idx] = ".raw"

    for file_type in file_types:
        if not os.path.isfile(os.path.join(data_path, file_name + file_type)):
            raise FileNotFoundError(f"{file_name}{file_type} not found in {data_path}")

    return True


def print_plot_limits(time_format="%Y-%m-%d %H:%M:%S", float_format="{:.2f}"):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    """
    Print the x and y limits of all axes in the current figure in a formatted way.

    Parameters
    ----------
    time_format : str
        Format to print the datetime objects. Default is '%Y-%m-%d %H:%M:%S'.
    float_format : str
        Format to print the float objects. Default is '{:.2f}'.

    Returns
    -------
    None
    """
    # Get the current figure
    fig = plt.gcf()

    # Get all axes in the current figure
    axs = fig.get_axes()

    for i, ax in enumerate(axs):
        # Get the x and y limits
        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()

        # Check if the x-axis uses a DateFormatter
        if isinstance(ax.xaxis.get_major_formatter(), mdates.DateFormatter):
            # Convert the x limits to a more human-readable format
            x_limits = tuple(
                datetime.strftime(
                    mdates.num2date(x),
                    time_format,
                )
                for x in x_limits
            )
            # Print the limits in a formatted way
            print(f"axs[{i}].set_xlim(pd.to_datetime({x_limits}))")
        else:
            # Format the x limits
            x_limits_formatted = (
                float_format.format(x_limits[0]),
                float_format.format(x_limits[1]),
            )
            # Print the limits in a formatted way
            print(
                f"axs[{i}].set_xlim({x_limits_formatted[0]}, {x_limits_formatted[1]})"
            )

            # Format the y limits
        y_limits_formatted = (
            float_format.format(y_limits[0]),
            float_format.format(y_limits[1]),
        )
        print(f"axs[{i}].set_ylim({y_limits_formatted[0]}, {y_limits_formatted[1]})")


def beam_from_log_files(log_file_path, filename):
    """
    Extracts the beam from a measurement .log file and returns the beam name.

    Parameters
    ----------
    log_file_path : str
        Path to the log file. Must contain .log and .cor files.
    filename : str
        Name of the log file. No extension.

    Returns
    -------
    beam : Beam
        Beam name.
    """

    # Check that both the .log (and .cor?) files exist
    if not os.path.exists(os.path.join(log_file_path, filename + ".log")):
        raise FileNotFoundError(f"Log file {filename}.log not found in {log_file_path}")

    # Read the log file
    log_file = os.path.join(log_file_path, filename + ".log")
    with open(log_file, "r") as f:
        log = f.readlines()

        # Extract the beam parameters from the log file
        # Example last line of log file:
        # notes = <beam 'AboveNorthEastDock'(dist 2177.56)(agl nan)(coor nan, nan, nan)(ned nan, nan, nan)(pointing -42.1570, -4.3420)>

        # NOTE this is a very fragile implementation that is relying on the last line's format and the existence of the single quotes.
        last_line = log[-1]
        beam_name = last_line.split("'")[1]

    return beam_name


def unwrap_etalon_lists(etalons):
    """
    Unwrap the nested list of etalons into a flat string.
    This is used to pass the etalons as command line arguments.

    Parameters
    ----------
    etalons: list of list of int
        List of etalon pairs.

    Returns
    -------
    etalons_flat: str
        Flat string of etalons.

    """
    etalons_flat = [str(e) for sublist in etalons for e in sublist]
    etalons_flat = " ".join(etalons_flat)
    return etalons_flat


def compare_end_and_total(start_val, end_val, avg_val, total_val):
    """
    This function checks for conflicts in the averaging information based on the
    configured inputs.
    Parameters
    ----------
    start_val: int
        Start index for averaging
    end_val: int
        End index for averaging
    avg_val: int
        Number of IGs to average
    total_val: int
        Total number of averaged IGs that will be produced

    Returns
    -------
    Raises exception is conflicts are identified
    """

    test_val = int(((end_val - start_val) / avg_val) + 1)
    if (test_val != total_val) or (end_val < avg_val):
        raise Exception("Values for end index and total number are not compatible")


def extract_noise_value_from_transmission(x_wvn, transmission, band_noise):
    """
    This function calculates the absorbance noise associated with the input
    transmission spectrum over the range indicated by band_noise. This function
    replaces the need to run an actual fit to determine the absorbance noise as long
    as band_noise does not span an absorption feature.
    Parameters
    ----------
    x_wvn: numpy array
        Wavenumber array for transmission spectrum
    transmission: numpy array
        Transmission array of data being processed
    band_noise: list
        List of indices over which to determine the absorbance noise.

    Returns
    -------
    noise_value: standard deviation of the absorbance in the specified region
    """
    if x_wvn[0] > x_wvn[len(x_wvn) - 1]:
        # Flip the wavelength axis and the transmission data
        # This is important for the searchsorted function
        x_wvn = x_wvn[::-1]
        transmission = transmission[int(len(x_wvn) - 1) :: -1]

    noise_start = np.searchsorted(x_wvn, band_noise[0], "left") - 1
    noise_stop = np.searchsorted(x_wvn, band_noise[1], "right")

    # Fit a polynomial to the absorbance data in the noise region to subtract baseline structure
    x_axis_subset = x_wvn[noise_start:noise_stop]
    transmission_subset = transmission[noise_start:noise_stop]
    absorbance_subset = -np.log(transmission_subset)

    poly_vals = np.polyfit(x_axis_subset, absorbance_subset, 3)
    p_absorbance = np.poly1d(poly_vals)
    noise_value = np.std(absorbance_subset - p_absorbance(x_axis_subset))
    return noise_value
