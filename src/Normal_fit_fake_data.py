import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc  # Import complementary error function from SciPy
from hapi import *

# Initialize HAPI
db_begin('data')

# Define a fake methane dataset
# Fake frequencies in cm^-1
frequencies = np.linspace(3000, 3050, 500)
# Simulated intensities (Voigt-like peaks)
intensities = (
    1.5 * np.exp(-0.5 * ((frequencies - 3015) / 1.2) ** 2) +
    0.8 * np.exp(-0.5 * ((frequencies - 3030) / 0.8) ** 2) +
    np.random.normal(0, 0.05, size=frequencies.shape)  # Add noise
)

# Define a Voigt function (simplified approximation for fitting)
def voigt_profile(f, amp, center, sigma, gamma):
    z = ((f - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(np.exp(-z**2) * erfc(-1j * z)) / (sigma * np.sqrt(2 * np.pi))

# Function to fit multiple Voigt peaks
def multi_voigt(f, *params):
    num_peaks = len(params) // 4  # Each peak has 4 parameters
    result = np.zeros_like(f)
    for i in range(num_peaks):
        amp, center, sigma, gamma = params[i * 4: (i + 1) * 4]
        result += voigt_profile(f, amp, center, sigma, gamma)
    return result

# Initial guess for fitting parameters (2 peaks)
initial_guess = [
    1.5, 3015, 1.2, 0.5,  # First peak: amp, center, sigma, gamma
    0.8, 3030, 0.8, 0.5   # Second peak
]

# Fit the data
popt, pcov = curve_fit(multi_voigt, frequencies, intensities, p0=initial_guess)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(frequencies, intensities, label='Simulated Data', linestyle='--', marker='o', markersize=2)
plt.plot(frequencies, multi_voigt(frequencies, *popt), label='Fit', linewidth=2)
plt.xlabel('Frequency (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('Methane Spectrum Fit')
plt.legend()
plt.grid()
plt.show()

# Display fitted parameters
for i in range(len(popt) // 4):
    amp, center, sigma, gamma = popt[i * 4: (i + 1) * 4]
    print(f"Peak {i + 1}: Amplitude={amp:.2f}, Center={center:.2f} cm⁻¹, Sigma={sigma:.2f}, Gamma={gamma:.2f}")
