import pytest
import numpy as np
from unittest.mock import Mock, patch
from my_spectral_analysis import (
    load_config_from_directory,
    setup_model_with_config,
    fit_data_from_directory,
)


# 1. Test loading configuration
def test_load_config_from_directory(tmpdir):
    config_dir = tmpdir.mkdir("configs")
    config_file = config_dir.join("test_config.json")
    config_file.write('{"fitting": {"band_fit": [6000, 6200]}}')

    config = load_config_from_directory(str(config_dir))
    assert config["fitting"]["band_fit"] == [6000, 6200]


# 2. Test setup model with configuration (mocking the model setup function)
def test_setup_model_with_config():
    mock_model_func = Mock(return_value=("mock_model", "mock_parameters"))
    config = {"fitting": {"some_param": 123}}

    model, params = setup_model_with_config(config, mock_model_func)

    mock_model_func.assert_called_once_with(config)
    assert model == "mock_model"
    assert params == "mock_parameters"


# 3. Test fit_data_from_directory
@patch("my_spectral_analysis.pld.trim_bandwidth", return_value=(np.array([1, 2, 3]), np.array([0.8, 0.6, 0.4])))
@patch("my_spectral_analysis.pld.calc_cepstrum", return_value=np.array([0.1, 0.2, 0.3]))
@patch("my_spectral_analysis.pld.weight_func", return_value=np.array([1, 1, 1]))
@patch("my_spectral_analysis.Model.fit", return_value="mock_fit_result")
def test_fit_data_from_directory(
        mock_fit,
        mock_weight_func,
        mock_calc_cepstrum,
        mock_trim_bandwidth,
        tmpdir
):
    # Set up temporary config directory
    config_dir = tmpdir.mkdir("configs")
    config_file = config_dir.join("test_config.json")
    config_file.write('{"fitting": {"band_fit": [6000, 6200], "baseline": 0, "etalons": []}}')

    # Define dummy data and mocks
    x_wvn = np.array([6000, 6100, 6200])
    transmission = np.array([0.9, 0.8, 0.7])
    mock_model_func = Mock(return_value=(Mock(Model), Mock()))

    # Call function
    fit_result = fit_data_from_directory(
        x_wvn,
        transmission,
        str(config_dir),
        mock_model_func,
        verbose=True
    )

    # Assertions
    assert fit_result == "mock_fit_result"
    mock_trim_bandwidth.assert_called_once_with(x_wvn, transmission, [6000, 6200])
    mock_calc_cepstrum.assert_called_once()
    mock_weight_func.assert_called_once_with(3, 0, [])
    mock_model_func.assert_called_once_with({"fitting": {"band_fit": [6000, 6200], "baseline": 0, "etalons": []}})
