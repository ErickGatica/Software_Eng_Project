import pytest
from unittest.mock import patch, MagicMock
import os
import matplotlib as mpl
from pldspectrapy.config_handling import load_config_json

# Move up two directories
os.chdir(os.path.abspath(os.path.join(os.getcwd(), "..", "..")))


# Test configure_plots
def test_configure_plots():
    from Normie_fitting import configure_plots

    configure_plots()

    # Check if plot parameters are updated correctly
    assert mpl.rcParams["backend"] == "TkAgg"
    assert mpl.rcParams["pdf.fonttype"] == 42
    assert mpl.rcParams["ps.fonttype"] == 42
    assert mpl.rcParams["font.family"] == "Arial"
    assert plt.rcParams["figure.autolayout"] is True
    assert plt.rcParams["lines.linewidth"] == 0.65
    assert plt.rcParams["mathtext.default"] == "regular"


# Test load_configuration
def test_load_configuration():
    from Normie_fitting import load_configuration

    # Mock the load_config_json function
    mock_config = {"test": "value"}
    with patch(
        "pldspectrapy.config_handling.load_config_json", return_value=mock_config
    ) as mock_load:
        config_path = "test_config.json"
        result = load_configuration(config_path)

        mock_load.assert_called_once_with(config_path)
        assert result == mock_config


# Test initialize_hapi_db
def test_initialize_hapi_db():
    from Normie_fitting import initialize_hapi_db

    # Mock the HAPI initialization function
    mock_config = {
        "fitting": {"simulation_backend": "not_gaas"},
        "input": {"linelist_path": "mock_path"},
    }
    with patch("pldspectrapy.td_support.initialize_hapi_db") as mock_hapi:
        initialize_hapi_db(mock_config)
        mock_hapi.assert_called_once_with("mock_path")


# Test process_file
def test_process_file():
    from Normie_fitting import process_file

    # Mock functions and objects used in process_file
    mock_config = {"input": {"fit_plots": False}, "output": {}}
    filepath = "test_file.cor"

    with patch("os.path.basename", return_value="test_file"), patch(
        "pldspectrapy.open_daq_files", return_value=MagicMock()
    ) as mock_open_daq, patch(
        "pldspectrapy.td_support.create_spectrum", return_value=([], [])
    ) as mock_create_spectrum, patch(
        "pldspectrapy.config_handling.generate_output_and_save",
        return_value=MagicMock(),
    ) as mock_generate_output, patch(
        "pldspectrapy.plotting_tools.plot_fit_td"
    ), patch(
        "pldspectrapy.plotting_tools.plot_fit_freq"
    ):

        process_file(filepath, mock_config)

        mock_open_daq.assert_called_once_with(filepath)
        mock_create_spectrum.assert_called()
        mock_generate_output.assert_called()


# Test plot_results
def test_plot_results():
    from Normie_fitting import plot_results

    mock_fit = MagicMock()
    mock_config = {}

    with patch("pldspectrapy.plotting_tools.plot_fit_td") as mock_plot_td, patch(
        "pldspectrapy.plotting_tools.plot_fit_freq"
    ) as mock_plot_freq:

        plot_results(mock_fit, mock_config)

        mock_plot_td.assert_called_once_with(mock_fit, mock_config)
        mock_plot_freq.assert_called_once_with(mock_fit, mock_config)


# Test main
def test_main():
    from Normie_fitting import main

    mock_config = {"input": {"fit_plots": False}, "output": {}}
    mock_file_list = ["file1.cor", "file2.cor"]

    with patch("Normie_fitting.configure_plots"), patch(
        "Normie_fitting.load_configuration", return_value=mock_config
    ), patch("Normie_fitting.initialize_hapi_db"), patch(
        "glob.glob", return_value=mock_file_list
    ), patch(
        "Normie_fitting.process_file"
    ) as mock_process:

        main()

        mock_process.assert_any_call("file1.cor", mock_config)
        mock_process.assert_any_call("file2.cor", mock_config)
