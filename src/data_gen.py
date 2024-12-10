import numpy as np
import pandas as pd  # To save data as CSV
import os

# Generate a fake methane dataset
frequencies = np.linspace(3000, 3050, 500)
intensities = (
    1.5 * np.exp(-0.5 * ((frequencies - 3015) / 1.2) ** 2) +
    0.8 * np.exp(-0.5 * ((frequencies - 3030) / 0.8) ** 2) +
    np.random.normal(0, 0.05, size=frequencies.shape)  # Add noise
)

# Save the data to a CSV file
output_file = 'methane_spectrum.csv'
data = pd.DataFrame({'Frequency': frequencies, 'Intensity': intensities})
data.to_csv(output_file, index=False)

print(f"Data generated and saved to {os.path.abspath(output_file)}")
