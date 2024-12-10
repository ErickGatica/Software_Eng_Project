import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz  # Use Faddeeva function for Voigt profile

# Define a Voigt function using wofz
def voigt_profile(f, amp, center, sigma, gamma):
    z = ((f - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

# Function to fit multiple Voigt peaks
def multi_voigt(f, *params):
    num_peaks = len(params) // 4  # Each peak has 4 parameters
    result = np.zeros_like(f)
    for i in range(num_peaks):
        amp, center, sigma, gamma = params[i * 4: (i + 1) * 4]
        result += voigt_profile(f, amp, center, sigma, gamma)
    return result

# Load the lookup table
lookup_table_file = "methane_absorption_multiple_conditions.csv"
data = pd.read_csv(lookup_table_file)

# Extract relevant data (e.g., for a specific mole fraction and temperature)
mole_fraction = 0.5  # Example mole fraction
temperature = 297  # Example temperature (K)
filtered_data = data[
    (data["Mole Fraction"] == mole_fraction) &
    (data["Temperature (K)"] == temperature)
]

# Ensure data is sorted by wavenumber
filtered_data = filtered_data.sort_values("Wavenumber (cm⁻¹)")

# Extract wavenumber and absorption coefficient
frequencies = filtered_data["Wavenumber (cm⁻¹)"].values
intensities = filtered_data["Absorption Coefficient"].values

# Initial guess for fitting parameters (e.g., 2 peaks)
initial_guess = [
    1.5, 3015, 1.2, 0.5,  # First peak: amp, center, sigma, gamma
    0.8, 3030, 0.8, 0.5   # Second peak
]

# Perform the fit
popt, pcov = curve_fit(multi_voigt, frequencies, intensities, p0=initial_guess)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(frequencies, intensities, label="Lookup Table Data", linestyle="--", marker="o", markersize=2)
plt.plot(frequencies, multi_voigt(frequencies, *popt), label="Fit", linewidth=2)
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Absorption Coefficient")
plt.title(f"Methane Spectrum Fit (Mole Fraction {mole_fraction}, Temp {temperature} K)")
plt.legend()
plt.grid()
plt.show()

# Display fitted parameters
for i in range(len(popt) // 4):
    amp, center, sigma, gamma = popt[i * 4: (i + 1) * 4]
    print(f"Peak {i + 1}: Amplitude={amp:.2e}, Center={center:.2f} cm⁻¹, Sigma={sigma:.2f}, Gamma={gamma:.2f}")
