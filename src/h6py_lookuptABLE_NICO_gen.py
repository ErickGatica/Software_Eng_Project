import h5py
import numpy as np
import itertools
from hapi import db_begin, absorptionCoefficient_Lorentz

# Step 1: Initialize HAPI Database
db_begin('hapi_data')

# Step 2: Define parameter ranges for the lookup table
temperature_range = np.linspace(200, 2000, 10)  # Temperature (K)
mole_fraction_range = np.linspace(0, 1, 10)  # Mole fraction
shift_range = np.linspace(-0.2, 0.2, 5)  # Shift range

# Step 3: Prepare HDF5 storage
h5_file = 'lookup_table.h5'

with h5py.File(h5_file, 'w') as h5f:
    # Create datasets to store parameters and absorption coefficients
    temp_ds = h5f.create_dataset('Temperature', data=temperature_range)
    mole_frac_ds = h5f.create_dataset('MoleFraction', data=mole_fraction_range)
    shift_ds = h5f.create_dataset('Shift', data=shift_range)

    # Placeholder dataset for absorption coefficients
    # Dimensions: [Temperature, MoleFraction, Shift, Absorption]
    absorption_shape = (
    len(temperature_range), len(mole_fraction_range), len(shift_range), int((2000 - 1000) / 0.1) + 1)
    absorption_ds = h5f.create_dataset('AbsorptionCoefficients', shape=absorption_shape, dtype=np.float64)

    # Step 4: Fill the dataset
    for i, (temp, mole_frac, shift) in enumerate(
            itertools.product(temperature_range, mole_fraction_range, shift_range)):
        # Compute absorption coefficients
        absorption = absorptionCoefficient_Lorentz(
            'CH4', mole_frac, temp, 1000, 2000, step=0.1
        )

        # Determine indices for storage
        t_idx = np.where(temperature_range == temp)[0][0]
        m_idx = np.where(mole_fraction_range == mole_frac)[0][0]
        s_idx = np.where(shift_range == shift)[0][0]

        # Store absorption coefficients in the dataset
        absorption_ds[t_idx, m_idx, s_idx, :] = absorption
