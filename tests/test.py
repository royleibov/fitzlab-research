import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft, fft

# Load the frequency domain data from the .npy files
comb_amplitudes_dBm = np.load('trace_matrix_baseline.npy')[0, :]
comb_frequencies = np.load('freq_matrix.npy')[0, :]

# Convert amplitude from dBm to linear scale
comb_amplitudes_linear = 10 ** ((comb_amplitudes_dBm - 30) / 10)

# Assign random phase (since phase is likely unknown)
phase = np.random.uniform(0, 2 * np.pi, len(comb_frequencies))

# Construct the complex signal in the frequency domain
freq_domain_signal = comb_amplitudes_linear * np.exp(1j * phase)

# Perform inverse FFT to get the time-domain signal
time_domain_signal = ifft(freq_domain_signal)

# Calculate frequency resolution (delta_f)
delta_f = comb_frequencies[1] - comb_frequencies[0]

# Determine total time duration T (from Nyquist theorem)
T = 1 / delta_f  # Total time in seconds

# Number of points in the frequency domain (which equals number of points in the time domain)
N = len(comb_frequencies)

# Time resolution (delta_t)
delta_t = T / N  # Time step

# Create the time vector
time_vector = np.linspace(0, T, N)

# Plot the time-domain signal (real part)
plt.figure(figsize=(10, 6))
plt.plot(time_vector, np.real(time_domain_signal))
plt.title("Time-Domain Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Now, perform FFT on the time-domain signal to reconstruct the frequency domain
reconstructed_freq_domain_signal = fft(time_domain_signal)

# Compute the amplitude from the reconstructed frequency domain signal
reconstructed_amplitudes = np.abs(reconstructed_freq_domain_signal)

# Convert reconstructed amplitudes to dBm for comparison
reconstructed_amplitudes_dBm = 10 * np.log10(reconstructed_amplitudes) + 30

# Plot the original frequency domain data vs reconstructed frequency domain data
plt.figure(figsize=(10, 6))
plt.plot(comb_frequencies, comb_amplitudes_dBm, label="Original Frequency Domain")
plt.plot(comb_frequencies, reconstructed_amplitudes_dBm, label="Reconstructed Frequency Domain", linestyle='--')
plt.title("Original vs Reconstructed Frequency Domain")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude (dBm)")
plt.legend()
plt.grid(True)
plt.show()
