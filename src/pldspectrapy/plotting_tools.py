import os

import numpy as np
from matplotlib import pyplot as plt
from .td_support import weight_func, compute_data_minus_baseline


# TODO: Refactor so that plotting functions are seperate and just take in arrays
#  This will allow us to reduce redundancy and make the code more modular
#  (i.e. plotting from fit objects vs plotting from arrays)


def plot_fit_td(fit_obj, config_dict):
    """Plot the time domain fit results

    Parameters
    ----------
    fit_obj : lmfit Model results object
        The results object created from running lmfit.Model.fit()
    config_dict : dictionary
        Configuration dictionary for setting fit parameters

    Returns
    -------
    None
        Saves the plot to the specified path
    """
    # TODO: Add plotting options to the config file? We already pass the config_dict to the plotting functions

    Fit = fit_obj

    plt.figure()
    plt.plot(Fit.data, label="data")
    plt.plot(Fit.best_fit, label="fit")
    plt.plot(Fit.data - Fit.best_fit, label="residual")
    plt.plot(Fit.weights, label="weights")

    plt.legend()
    plt.show()
    if config_dict["input"]["save_plot"]:
        plt.savefig(
            os.path.join(
                config_dict["input"]["plot_path"],
                config_dict["input"]["plot_name"] + "_td.png",
            )
        )


def plot_fit_freq(fit_obj, config_dict):
    """Plot the frequency domain fit results

    Parameters
    ----------
    fit_obj : lmfit Model results object
        The results object created from running lmfit.Model.fit()
    config_dict : dictionary
        Configuration dictionary for setting fit parameters

    Returns
    -------
    None
        Saves the plot to the specified path
    """

    data_less_bl_time_domain = compute_data_minus_baseline(
        data=fit_obj.data, weight=fit_obj.weights, fit=fit_obj.best_fit
    )

    data_less_bl_freq_domain = np.real(np.fft.rfft(data_less_bl_time_domain))

    model_frequency_domain = np.real(np.fft.rfft(fit_obj.best_fit))
    residual_frequency_domain = np.real(np.fft.rfft(fit_obj.residual))
    x_wvn_trimmed = fit_obj.userkws["xx"]

    fig, axs = plt.subplots(2, 1, sharex="col", gridspec_kw={"height_ratios": [3, 1]})

    axs[0].plot(x_wvn_trimmed, data_less_bl_freq_domain, label="data")
    axs[0].plot(x_wvn_trimmed, model_frequency_domain, label="fit")

    axs[1].plot(x_wvn_trimmed, residual_frequency_domain, label="residual")
    axs[0].set_ylabel("Absorbance")
    axs[0].legend()
    axs[1].set_ylabel("Residual")
    axs[1].set_xlabel("Wavenumber ($cm^{-1}$)")

    # despine
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if config_dict["input"]["save_plot"]:
        plt.savefig(
            os.path.join(
                config_dict["input"]["plot_path"],
                config_dict["input"]["plot_name"] + "_fd.png",
            )
        )
    plt.show()

def plot_transmission(wvn, transmission):
    """Plot the transmission spectrum

    Parameters
    ----------
    wvn : numpy array
        Wavenumber axis of transmission spectrum
    transmission : array
        Transmission spectrum

    Returns
    -------
    None
        Saves the plot to the specified path
    """

    plt.figure()
    plt.plot(wvn, transmission, label="Transmission")

    plt.legend()


def generate_tutorial_fit_plots(x_wvn_full, transmission, config_variables, Fit):
    """
    Generates a series of plots to illustrate the fitting process

    Parameters
    ----------
    x_wvn_full : numpy array
        wavenumber array
    transmission : numpy array
        transmission array
    config_variables : dictionary
        contains all the values parsed from the config file
    Fit : lmfit.Model object
        fit object containing the model and parameters. Can be used to extract fit parameters and uncertainties

    Returns
    -------
    None

    """
    # # Show the raw transmission spectrum, the selected band, pull it out and then show the cepstrum with the weight function applied with shading
    # # Use subplots
    # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    # axs[0, 0].plot(x_wvn_full, transmission)
    # axs[0, 0].plot(
    #     x_wvn_full[config_variables["fitting"]["band_fit"]],
    #     transmission[config_variables["fitting"]["band_fit"]],
    # )
    # axs[0, 0].set_title("Raw Transmission Spectrum")
    # axs[0, 0].set_xlabel("Wavenumber (cm-1)")
    # axs[0, 0].set_ylabel("Transmission")
    #
    # # Plot the absorbance of the un-baseline corrected data
    # axs[0, 1].plot(x_wvn_full, -np.log(transmission))
    # axs[0, 1].plot(
    #     x_wvn_full[config_variables["fitting"]["band_fit"]],
    #     -np.log(transmission[config_variables["fitting"]["band_fit"]]),
    # )
    # axs[0, 1].set_title("Absorbance Spectrum")
    # axs[0, 1].set_xlabel("Wavenumber (cm-1)")
    #
    # # Plot the cepstrum
    # axs[1, 0].plot(x_wvn_full, pld.calc_cepstrum(transmission))
    # axs[1, 0].set_title("Cepstrum")
    # axs[1, 0].set_xlabel("Wavenumber (cm-1)")
    #
    #
    # # Plot the baseline-removed absorbance
    return None


def generate_fit_plots(results_directory, filename, baseline=None, etalons=None):
    """
    Generates a subplot of time domain and frequency domain fit results
    From LPT-style results directory (i.e. has directories:
        debug
        fit_logs
        fit_plots
        output
        residual_etalons_results
        residuals
        y_td
    )

    Parameters
    ----------
    results_directory : str
        Path to the results directory
    filename : str
        Name of the file to plot

    Returns
    -------
    fig : matplotlib figure
        The figure object
    axs : matplotlib axis
        The axis object

    """

    # make sure that the directories that we will need exist
    residual_path = os.path.join(
        results_directory, "residuals", f"{filename}_residual.csv"
    )
    cepstrum_path = os.path.join(results_directory, "y_td", f"{filename}_y_td.csv")

    if not os.path.exists(residual_path):
        raise FileNotFoundError(f"Residual file not found at {residual_path}")
    if not os.path.exists(cepstrum_path):
        raise FileNotFoundError(f"Cepstrum file not found at {cepstrum_path}")

    # Load the cepstrum from the y_td directory
    data_cepstrum = np.loadtxt(cepstrum_path)
    residual_cepstrum = np.loadtxt(residual_path)

    # Make a weight function to visualize the data minus baseline
    if baseline is None and etalons is None:
        weight_array = np.ones(len(data_cepstrum))
    else:
        weight_array = weight_func(
            int(len(data_cepstrum) / 2 + 1), baseline, etalons=etalons
        )

    # Compute the cepstrum fit  from the data and residual
    fit_cepstrum = data_cepstrum - residual_cepstrum

    # Compute the data minus baseline cepstrum
    # TODO: put this in a function (used many places)
    data_minus_baseline_cepstrum = compute_data_minus_baseline(
        data=data_cepstrum, weight=weight_array, fit=fit_cepstrum
    )

    # Compute the frequency domain information from the cepstrum
    data_freq = np.fft.rfft(data_cepstrum)
    fit_freq = np.fft.rfft(fit_cepstrum)
    data_minus_baseline_frequency = np.fft.rfft(data_minus_baseline_cepstrum)
    residual_freq = data_minus_baseline_frequency - fit_freq

    fig, axs = plt.subplots(
        2,
        2,
        gridspec_kw={"height_ratios": [3, 1], "width_ratios": [1, 2]},
        sharex="col",
    )

    # despine top and right on all plots
    for ax in axs.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Plot the time domain data
    axs[0, 0].plot(data_cepstrum, label="data")
    axs[0, 0].plot(fit_cepstrum, label="fit")
    axs[0, 0].legend()

    axs[1, 0].plot(residual_cepstrum, label="residual", color="darkgrey")
    axs[1, 0].legend()

    # Plot the frequency domain data
    axs[0, 1].plot(data_minus_baseline_frequency, label="data minus baseline")
    axs[0, 1].plot(fit_freq, label="fit")
    axs[0, 1].legend()

    axs[1, 1].plot(residual_freq, label="residual", color="darkgrey")
    axs[1, 1].legend()

    return fig, axs
