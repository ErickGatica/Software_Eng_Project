def get_spectral_data_from_csv(lookup_table, config_variables, x_wvn, model_name="ch4_1"):
    """
    Retrieve spectral data from CSV lookup table.

    Parameters
    ----------
    lookup_table : pd.DataFrame
        DataFrame containing the lookup table.
    config_variables : dict
        Configuration dictionary with fitting parameters.
    x_wvn : numpy array
        Wavenumber array.
    model_name : str, optional
        The model setup name (e.g., "ch4_1").

    Returns
    -------
    spectral_data : numpy array
        The spectral data matching the conditions.
    """
    def get_model_value(config, model_name, parameter):
        try:
            return config["model_setup"][model_name][parameter]["value"]
        except KeyError:
            raise ValueError(f"{parameter} not found in model {model_name}.")

    temperature = get_model_value(config_variables, model_name, "temperature")
    mole_fraction = get_model_value(config_variables, model_name, "molefraction")

    matching_rows = lookup_table[
        (lookup_table["Temperature (K)"] == temperature) &
        (lookup_table["Mole Fraction"] == mole_fraction)
    ]

    if matching_rows.empty:
        raise ValueError(f"No matching data for T={temperature}, mole_fraction={mole_fraction}.")

    # Interpolate data for the provided wavenumbers
    spectral_data = np.interp(
        x_wvn, matching_rows["Wavenumber (cm⁻¹)"], matching_rows["Absorption Coefficient"]
    )
    return spectral_data
