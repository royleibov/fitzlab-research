import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq, fftshift, ifftshift

# 1. Load the data
comb_amplitudes_db = np.load('trace_matrix_baseline.npy')[0, :]
comb_frequencies = np.load('freq_matrix.npy')[0, :]

# Convert amplitudes from dB to linear scale
comb_amplitudes_linear = 10 ** (comb_amplitudes_db / 20)

plt.figure(figsize=(10, 6))
plt.plot(comb_frequencies, comb_amplitudes_db, linestyle='-')
plt.title('Amplitudes vs. Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 2. Data Preprocessing

# Remove any NaN or Inf values
valid_indices = np.isfinite(comb_frequencies) & np.isfinite(comb_amplitudes_db)
comb_frequencies = comb_frequencies[valid_indices]
comb_amplitudes_db = comb_amplitudes_db[valid_indices]

# Sort the data by frequency
sorted_indices = np.argsort(comb_frequencies)
comb_frequencies = comb_frequencies[sorted_indices]
comb_amplitudes_db = comb_amplitudes_db[sorted_indices]

# 3. Interpolate onto a uniform grid
num_points = len(comb_frequencies)
print(f"Number of points: {num_points}")
freq_uniform = np.linspace(comb_frequencies[0], comb_frequencies[-1], num_points)

# Interpolate amplitudes onto the uniform grid
amplitude_interp = interp1d(comb_frequencies, comb_amplitudes_db, kind='cubic', fill_value="extrapolate")
comb_amplitudes_uniform = amplitude_interp(freq_uniform)

# 4. Shift Frequencies to Baseband
f_min = freq_uniform[0]
f_max = freq_uniform[-1]
F_span = f_max - f_min
f_c = (f_max + f_min) / 2  # Carrier frequency

# Shift frequencies to baseband
freq_baseband = freq_uniform - f_c

# Create the complex spectrum with zero phase
spectrum = comb_amplitudes_uniform * np.exp(1j * 0)

# Shift the spectrum to center it around zero frequency
spectrum_shifted = fftshift(spectrum)

# 5. Perform IFFT on the shifted spectrum
time_signal = np.fft.ifft(spectrum_shifted)

# Compute the corresponding time vector
freq_spacing = freq_uniform[1] - freq_uniform[0]
df = freq_spacing
print(f"df: {df}")
dt = 1 / (num_points * df)
print(f"dt: {dt}")
time_vector = np.arange(-num_points // 2, num_points // 2) * dt

# 6. Plot the Time-Domain Signal
plt.figure(figsize=(10, 6))
plt.plot(time_vector, np.real(time_signal))
plt.title('Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# 6.5. Save the Time-Domain Signal
np.save('time_signal.npy', time_signal)

# 7. Perform FFT to Get Back to Frequency Domain
# Perform FFT on the time-domain signal
spectrum_reconstructed = np.fft.fft(time_signal)

# Shift the spectrum back
spectrum_reconstructed_shifted = ifftshift(spectrum_reconstructed)

# Frequencies for plotting
fft_freqs = fftfreq(num_points, dt)
fft_freqs_shifted = fftshift(fft_freqs) + f_c  # Shift frequencies back to original range

# 8. Plot the Reconstructed Frequency Spectrum
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs_shifted, np.abs(spectrum_reconstructed_shifted), label='Reconstructed Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Frequency Spectrum')
plt.grid(True)
plt.show()
