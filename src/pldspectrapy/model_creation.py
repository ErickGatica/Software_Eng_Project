import lmfit

# import radis # Eli commented out on 12/11/23 when using Sean's data_quality_checks with the debugger

import pldspectrapy as pld

exclude_names = ["global", "db_name"]


def update_parameters(fit_dict, params):
    """
    Updates the parameters object with the parameters from the config file.
    This function should merge the functionality of update_nonglobal_params and
    update_global_parameters

    Parameters
    ----------
    fit_dict : dictionary
        contains all the values parsed from the config file

    params : lmfit.Parameters object
        parameters to be used in the fitting process. Should already have
        any global parameters added

    Returns
    -------
    params : lmfit.Parameters object
        parameters to be used in the fitting process
    """

    # Update the non-global parameters
    # real_molecules = [molecule for molecule in fit_dict.keys() if molecule != "global"]

    excluded_properties = ["db_name", "gaas_key", "gaas_db"]

    for molecule in fit_dict.keys():
        for property, property_values in fit_dict[molecule].items():
            # Skip the db_name or gaas_key property. This is explicitly passed to the
            # model and is not a parameter to be fit. This sill allows us to
            # specify the database for each molecule as a string in the config file
            if property in excluded_properties:
                continue

            # this skips if a global parameter is found. The expresion is set in
            # add_global_parameters() (i.e., pressure: "global" is an entry in a molecule)
            if property_values == "global":
                continue

            # Include the prefix for the parameter name
            param_name = f"{molecule}_{property}"

            # Update the parameters object using dictionary unpacking to set the values
            params[param_name].set(**property_values)

    return params


def update_scale_model_parameters(fit_dict, params):
    """
    Updates the parameters object with the parameters from the config file.
    This function should merge the functionality of update_nonglobal_params and
    update_global_parameters

    Parameters
    ----------
    fit_dict : dictionary
        contains all the values parsed from the config file

    params : lmfit.Parameters object
        parameters to be used in the fitting process. Should already have
        any global parameters added

    Returns
    -------
    params : lmfit.Parameters object
        parameters to be used in the fitting process
    """

    # Update the non-global parameters
    # real_molecules = [molecule for molecule in fit_dict.keys() if molecule != "global"]

    excluded_properties = [
        "db_name",
        "gaas_key",
        "gaas_db",
        "scale_key",
        "scale_models",
    ]

    for molecule in fit_dict.keys():
        for property, property_values in fit_dict[molecule].items():
            if property != "scalefactor":
                continue
            # Include the prefix for the parameter name
            param_name = f"{molecule}_{property}"

            # Update the parameters object using dictionary unpacking to set the values
            params[param_name].set(**property_values)

    return params


def setup_models(config_dict, print_params=False, absDB=None):
    """
    THIS IS THE MAIN FUNCTION TO BE CALLED FROM THE OUTSIDE

    Sets up the models to be used in the fitting process. Uses the lmfit
    package and td_support.spectra_single to create the models. Should
    allow for multispecies fitting and variable or fixed thermodynamic
    properties

    Parameters
    ----------
    config_dict : dictionary
        contains all the values parsed from the config file

    print_params: bool
        prints the params for inspection

    Returns
    -------
    model : lmfit.Model object
        model to be used in the fitting process

    parameters : lmfit.Parameters object
        parameters to be used in the fitting process
    """

    # The fitting portion of the config file
    fit_dict = config_dict["model_setup"]
    input_dict = config_dict["fitting"]  # The input portion of the config file

    # Validate the fit_dict
    validate_fit_dict(fit_dict)

    model_full = create_composite_model(
        fit_dict, model_name=input_dict["simulation_backend"]
    )  # This is where we point to spectra_single

    params = model_full.make_params()  # lmfit built-in function (Yeah OOP)

    # Are there any global parameters? If so, we need to add them to the params
    # TODO: extend this functionality to allow for multiple global parameters
    #   The goal would be that the user could specify different populations of
    #   molecules with different global parameters. (i.e., Nate-style temperature
    #   distribution fitting, all from the config file)

    if "global" in fit_dict.keys():
        global_keys_mapping = determine_global_params(fit_dict)

        # Add global parameters to the params
        params = add_global_parameters(global_keys_mapping, params)

    # Set up all parameters from the config file
    if "scale" not in input_dict["simulation_backend"]:
        params = update_parameters(fit_dict, params)
    else:
        params = update_scale_model_parameters(fit_dict, params)

    if print_params:
        params.pretty_print()

    return model_full, params


def determine_global_params(fit_dict):
    """Determines which parameters are to be set globally.

    Goal is that we can use the output dictionary to set the overall
    parameters object with expressions pointing to global parameters
    example usage:
    params['m1_c'].set(expr='global_c')
    or in a loop (actual usage):
        for key in global_keys_mapping:
            params[key].set(expr=global_keys_mapping[key])

    Parameters
    ----------
    fit_dict : dictionary
        dictionary containing all the values parsed from the config file
        example:
        'ch4':
            {
                "pressure": {"value": 0.95, "vary": False},
                "temperature": {"value": 290, "vary": False},
                "molefraction": {"value": 2e-6, "vary": True},
            },
        'co2':
            {...

    Returns
    -------
    global_keys_mapping : dict
        dictionary with keys of global parameters and values of the local
        parameter they are mapped to

    """
    global_key_mapping = {}

    for molecule, properties in fit_dict.items():
        for prop, prop_dict in properties.items():
            if prop == "db_name":
                continue

            if prop_dict == "global":
                # Add to the global key mapping dict.
                # Example:
                # global_key_mapping['ch4_molefraction'] = 'global_molefraction'
                key_without_suffix = prop.split("_")[0]
                global_key_mapping[
                    f"{molecule}_{key_without_suffix}"
                ] = f"global_{key_without_suffix}"

    return global_key_mapping


def update_nonglobal_params(fit_dict, params):
    """
    Updates the parameters object with the parameters from the config file
    that are not global parameters

    Parameters
    ----------
    fit_dict : dictionary
        contains all the values parsed from the config file
    params : lmfit.Parameters object
        parameters to be used in the fitting process. Should already have
        any global parameters added
    """
    real_molecules = [
        molecule for molecule in fit_dict.keys() if molecule not in exclude_names
    ]

    for molecule in real_molecules:
        for prop, prop_dict in fit_dict[molecule].items():
            # this skips if a global parameter is found
            # (i.e., pressure: "global" is an entry in a molecule)
            if prop_dict == "global":
                continue

            param_name = f"{molecule}_{prop}"

            # Update the parameters object using unpacking to set the values
            params[param_name].set(**prop_dict)

    return params


def add_global_parameters(global_keys_mapping, params):
    """
    Adds the global parameters to the params object and updates the expressions

    Parameters
    ----------
    global_keys_mapping : dict
        dictionary with keys of global parameters and values of the local parameter
        they are mapped to
        This is output from determine_global_params
    params : lmfit.Parameters object
        parameters to be used in the fitting process

    Returns
    -------
    params : lmfit.Parameters object
        parameters to be used in the fitting process with global parameters added

    """

    for key, expr in global_keys_mapping.items():
        # Create the global parameter in the params object
        params.add(expr)

        # Update the expression to point to the global parameter
        params.add(key, expr=expr)

    return params


def update_global_parameters(fit_dict, params):
    # Using the "global" key in the fit_dict, update the global parameters in the
    # params object. This is used to add initial values and bounds to the global
    # parameters per the config file

    for prop, prop_dict in fit_dict["global"].items():
        prop_with_prefix = "global_" + prop

        # Update the parameters object utilizing unpacking to set the values
        params[prop_with_prefix].set(**prop_dict)

    return params


def create_composite_model(fit_dict, model_name="hapi"):
    """
    Creates a composite model from the fit_dict

    Parameters
    ----------
    fit_dict : dictionary
        dictionary containing all the values parsed from the config file

    model_name: str
        alias for the backend model to use for fitting - currently supports "hapi",
        "hapi_sd", and "radis".

    Returns
    -------
    model_full : lmfit.CompositeModel object
        composite model to be used in the fitting process
    """
    # Set up the model for each molecule in the config file
    model_list = []
    molecules = fit_dict.keys()

    # Remove "global" from the list of molecules. These are not molecules,
    # they are global parameters
    molecules = [molecule for molecule in molecules if molecule not in exclude_names]

    # TODO Adding db_name to the model nominally worked except this expects a float or an
    #  actual variable rather than a string so an error arises during the model
    #  creation (but it does show up as a parameter...
    #  Maybe try passing some sort of identifier that maps to a dictionary or
    #  something that would allow the direct mapping we are hoping to accomplish with
    #  this update

    model_functions = {
        "hapi": pld.spectra_single,
        "hapi_sd": pld.spectra_sd,
        "radis": pld.spectra_single_radis,
        "gaas": pld.spectra_single_gaas,
        "scale_hapi": pld.spectra_scale_model,
        "hapi2": pld.spectra_single_hapi2,
    }

    for molecule in molecules:
        molecule_database_name = fit_dict[molecule][
            "db_name"
        ]  # Choosing to forego the "value" structure if possible

        model_setup_arguments = {
            "prefix": molecule + "_",
            "db_name": molecule_database_name,
        }

        # This will enable using GAAS for fast fitting of large/hot data
        if model_name == "gaas":
            model_setup_arguments["gaas_key"] = fit_dict[molecule]["gaas_key"]
            model_setup_arguments["absDB_dict"] = fit_dict[molecule]["gaas_db"]
        elif "scale" in model_name:
            model_setup_arguments["scale_key"] = fit_dict[molecule]["scale_key"]
            model_setup_arguments["scale_models"] = fit_dict[molecule]["scale_models"]

        if model_name in model_functions:
            model = lmfit.Model(model_functions[model_name], **model_setup_arguments, independent_vars=["xx"])

            model_list.append(model)
        else:
            raise ValueError(
                f"model_name must be {model_functions.keys()}" f"Recieved {model_name}"
            )

    # Combine all the models into a CompositeModel
    model_full = model_list[
        0
    ]  # Need to initialize the model_full object as the right type
    for model in model_list[1:]:
        model_full += model
    return model_full


def validate_fit_dict(fit_dict):
    """
    Validates the fit_dict to make sure it has the correct format

    Parameters
    ----------
    fit_dict : dictionary
        dictionary containing all the values parsed from the config file

    Returns
    -------
    None

    """
    # TODO: This could get placed elsewhere, but right now if you specify a
    #  database name, there is not checking if it exists within the linelist directory.
    #  This could be a good place to add that check.
    #   Currently, it will fail with a KeyError in haip in getDefaultValuesForXsect

    # Ensure that each molecule has the required properties and has a vary status

    for molecule, properties in fit_dict.items():
        if molecule == ("global") or ("db_names"):
            # Skip the global parameters, they are not molecules
            continue

        for prop in ["pressure", "temperature", "molefraction"]:
            if prop not in properties:
                raise ValueError(
                    f"Missing {prop} for {molecule} in fit_dict. All molecules must "
                    f"have pressure, temperature, and molefraction defined"
                )
            if "vary" not in properties[prop]:
                raise ValueError(
                    f"Missing vary status for {prop} for {molecule} in fit_dict. "
                    f"All molecules must have pressure, temperature, and molefraction "
                    f"defined"
                )

        # Ensure that the db_name is present
        if "db_name" not in properties:
            raise ValueError(
                f"Missing db_name for {molecule} in fit_dict. All molecules must have "
                f"a db_name defined"
            )

        # Ensure that the db_name is in the correct format. Expected format is
        # "db_name": {"value": "Value"}
        if not isinstance(properties["db_name"], dict):
            raise ValueError(
                f"db_name for {molecule} in fit_dict must be a dictionary "
                f"with a 'value'"
                f"key. \n Example format: 'db_name': {{'value': 'CH4_08'}}"
            )


def update_pathlength(params, input_dict):
    """
    Updates the pathlength in the params object based on the input_dict

    Parameters
    ----------
    params : lmfit.Parameters object
        parameters to be used in the fitting process
    input_dict : dictionary
        dictionary containing all the values parsed from the config file.
        This is under the heading "fitting" rather than "molcules"

    Returns
    -------
    params : lmfit.Parameters object
        parameters to be used in the fitting process with pathlength updated
    """

    # Update all the pathlengths for all molecules

    for param in params.values():
        if param.name.endswith("pathlength"):
            param.set(
                value=input_dict["pathlength"], vary=False
            )  # Normally don't want to vary the pathlength

    return params
