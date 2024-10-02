print("Importing necessary libraries...")
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, sesolve, sigmam
from scipy.fft import fft, fftfreq, fftshift, ifftshift, ifft
from tqdm import tqdm
from scipy.interpolate import interp1d
# import math
print("Imported necessary libraries")

# Number of TLS
N = 3  # Adjust N as needed

# System parameters
omega_0 = 5e9        # Base frequency of the TLS (5 GHz)
Omega_R = 0.1e9      # Rabi frequency (100 MHz)
n_max = 100          # Number of frequency components in the comb
Delta_omega = 10e6   # Comb spacing (10 MHz)
t_span = (0, 1e-6)   # Time span for the simulation (1 microsecond)
dt = 1e-9            # Time step (1 ns)
t_eval = np.arange(t_span[0], t_span[1], dt)
f_c = 0              # Carrier frequency

# Coupling strength between TLS
J = 1e6  # Coupling strength (1 MHz)

# Static electric field parameters
E_field = 100000.0  # Static electric field strength (V/m)
dipole_moment = 4.0  # Dipole moment of the TLS (in C·m, or alternatively, GHz·m)

# Plot the random frequency shifts
# plt.figure()
# plt.plot(t_eval, np.zeros_like(t_eval), 'k--', label='Base frequency')
# plt.step(shift_times, shift_magnitudes, color='red', label='Random shifts')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency Shift (Hz)')
# plt.title('Random Frequency Shifts')
# plt.legend()
# plt.grid(True)
# plt.show(block=False)

# Identity and Pauli operators
I = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()
sm = sigmam()

def tensor_operator(op_list):
    return tensor(op_list)

def operator_at_site(op, site, N):
    op_list = [I] * N
    op_list[site] = op
    return tensor_operator(op_list)

# Define omega_shifted function
def omega_shifted(t, args):
    omega_0 = args['omega_0']
    shift_times = args['shift_times']
    assert len(shift_times) == 1, "Only one shift time is supported"
    shift_magnitudes = args['shift_magnitudes']
    assert len(shift_magnitudes) == 1, "Only one shift magnitude is supported"
    # idx = np.searchsorted(shift_times, t, side='right') - 1
    # if idx >= 0:
    #     shift_applied = shift_magnitudes[idx]
    # else:
    #     shift_applied = 0.0
    # shift_applied = shift_magnitudes[0] if t >= shift_times[0] else 0.0
    shift_applied = 0.0
    
    # tqdm.write(f"Shift applied: {t}")
    return omega_0 + shift_applied

# Frequency comb function
def frequency_comb(t, args):
    n_max = args['n_max']
    Delta_omega = args['Delta_omega']
    omega_base = args['omega_0']
    Omega_R = args['Omega_R']
    delta_omega_t_over_2 = Delta_omega * t / 2
    numerator = np.sin((2 * n_max + 1) * delta_omega_t_over_2)
    denominator = np.sin(delta_omega_t_over_2)
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.isclose(denominator % (2 * np.pi), 0),
            np.cos(omega_base * t) * (2 * n_max + 1),
            np.cos(omega_base * t) * numerator / denominator
        )
    # result = np.sum([np.cos((omega_base + n * Delta_omega) * t) for n in range(-n_max, n_max + 1)], axis=0)
    return Omega_R * result

# Load experimental time-domain signal
print('Loading experimental time-domain signal...')
# # 1. Load the data
# comb_amplitudes_db = np.load('trace_matrix_baseline.npy')[0, :]
# comb_frequencies = np.load('freq_matrix.npy')[0, :]

# # Convert amplitudes from dB to linear scale
# # comb_amplitudes_linear = 10 ** (comb_amplitudes_db / 20)
# comb_amplitudes_linear = comb_amplitudes_db

# # 2. Data Preprocessing

# # Remove any NaN or Inf values
# valid_indices = np.isfinite(comb_frequencies) & np.isfinite(comb_amplitudes_linear)
# comb_frequencies = comb_frequencies[valid_indices]
# comb_amplitudes_linear = comb_amplitudes_linear[valid_indices]

# # Sort the data by frequency
# sorted_indices = np.argsort(comb_frequencies)
# comb_frequencies = comb_frequencies[sorted_indices]
# comb_amplitudes_linear = comb_amplitudes_linear[sorted_indices]

# # 3. Interpolate onto a uniform grid
# num_points = len(comb_frequencies)
# freq_uniform = np.linspace(comb_frequencies[0], comb_frequencies[-1], num_points)

# # Interpolate amplitudes onto the uniform grid
# amplitude_interp = interp1d(comb_frequencies, comb_amplitudes_linear, kind='cubic', fill_value="extrapolate")
# comb_amplitudes_uniform = amplitude_interp(freq_uniform)

# # 4. Shift Frequencies to Baseband
# f_min = freq_uniform[0]
# f_max = freq_uniform[-1]
# F_span = f_max - f_min
# f_c = (f_max + f_min) / 2  # Carrier frequency

# # Shift frequencies to baseband
# freq_baseband = freq_uniform - f_c

# # Create the complex spectrum with zero phase
# spectrum = comb_amplitudes_uniform * np.exp(1j * 0)

# # Shift the spectrum to center it around zero frequency
# spectrum_shifted = fftshift(spectrum)

# # 5. Perform IFFT on the shifted spectrum
# time_signal = np.fft.ifft(spectrum_shifted)

# # Compute the corresponding time vector
# freq_spacing = freq_uniform[1] - freq_uniform[0]
# df = freq_spacing
# dt = 1 / (num_points * df)
# time_vector = np.arange(0, num_points) * dt

# Load the frequency domain data from the .npy files
# comb_amplitudes_dBm = np.load('trace_matrix_baseline.npy')[0, :]
# comb_frequencies = np.load('freq_matrix.npy')[0, :]

# # Convert amplitude from dBm to linear scale
# comb_amplitudes_linear = 10 ** ((comb_amplitudes_dBm - 30) / 10)

# # Assign random phase (since phase is likely unknown)
# phase = np.random.uniform(0, 2 * np.pi, len(comb_frequencies))

# # Construct the complex signal in the frequency domain
# freq_domain_signal = comb_amplitudes_linear * np.exp(1j * phase)

# # Perform inverse FFT to get the time-domain signal
# time_domain_signal = ifft(freq_domain_signal)

# # Calculate frequency resolution (delta_f)
# delta_f = comb_frequencies[1] - comb_frequencies[0]

# # Determine total time duration T (from Nyquist theorem)
# T = 1 / delta_f  # Total time in seconds

# # Number of points in the frequency domain (which equals number of points in the time domain)
# num_freqs = len(comb_frequencies)

# # Time resolution (delta_t)
# delta_t = T / num_freqs  # Time step

# # Create the time vector
# time_vector = np.linspace(0, T, num_freqs)

# t_eval = time_vector

# time_signal = np.load('time_signal.npy')
# comb_frequencies = np.load('comb_frequencies.npy')

# # Create a uniform frequency grid
# num_points = len(comb_frequencies)
# freq_uniform = np.linspace(comb_frequencies[0], comb_frequencies[-1], num_points)

# # Compute the frequency comb over t_eval
# # Compute the corresponding time vector
# len_time = len(time_signal)
# F_span = freq_uniform[-1] - freq_uniform[0]

# dt = (len_time - 1) / (len_time * F_span)
# # Determine the order of magnitude of dt
# exponent = int(math.floor(math.log10(abs(dt))))
# mantissa = dt / (10 ** exponent)

# # Decide on the number of significant figures
# significant_figures = 2  # Adjust as needed
# mantissa_rounded = round(mantissa, significant_figures - 1)

# # Reconstruct the rounded dt
# dt = mantissa_rounded * (10 ** exponent)

# t_span = (0, len_time*dt)   # Time span for the simulation (1 microsecond)
# t_eval = np.arange(t_span[0], t_span[1], dt)

# # Apply a Hanning window to smooth the time-domain signal and reduce high-frequency artifacts
# window = np.hanning(len(time_domain_signal))
# smoothed_time_domain_signal = np.real(time_domain_signal) * window

# # Use smoothed signal for further calculations
# signal_interpolator = smoothed_time_domain_signal


# signal_interpolator = interp1d(t_eval, np.real(time_signal), kind='cubic', fill_value="extrapolate")
# signal_interpolator = signal_interpolator(t_eval)
# signal_interpolator = Cubic_Spline(t_eval, np.real(time_signal))
# signal_interpolator = time_domain_signal#time_signal

# Generate random frequency shifts
print('Generating random frequency shifts...')
num_shifts = 3  # Number of random shifts
max_shift = 1e6  # Maximum shift magnitude (1 MHz)
shift_times = np.sort(np.random.choice(t_eval, num_shifts, replace=False))
shift_magnitudes = np.random.uniform(-max_shift, max_shift, num_shifts)
print('Done!')

# def experimental_frequency_comb(t, args):
#     Omega_R = args['Omega_R']
#     # time_signal = args['time_signal']
#     return Omega_R * signal_interpolator

# # Compute the frequency comb over t_eval
# print('Computing frequency comb values for plotting...')
# comb_values = np.array([frequency_comb(t, 
#                                        {
#                                         'n_max': n_max,
#                                         'Delta_omega': Delta_omega,
#                                         'omega_0': omega_0,
#                                         'Omega_R': Omega_R
#                                         }) for t in t_eval])

# fft_comb = fft(signal_interpolator)
# fft_freqs_comb = fftfreq(len(t_eval), dt)

# # Now, perform FFT on the time-domain signal to reconstruct the frequency domain
# reconstructed_freq_domain_signal = fft(smoothed_time_domain_signal)

# # Compute the amplitude from the reconstructed frequency domain signal
# reconstructed_amplitudes = np.abs(reconstructed_freq_domain_signal)

# # Convert reconstructed amplitudes to dBm for comparison
# reconstructed_amplitudes_dBm = 10 * np.log10(reconstructed_amplitudes) + 30

# # Plot the time-domain signal
# plt.figure(figsize=(10, 6))
# plt.plot(t_eval, signal_interpolator)
# plt.title('Time-Domain Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.grid(True)
# plt.show()

# # Plot the frequency comb
# plt.figure(figsize=(10, 6))
# plt.plot(comb_frequencies, reconstructed_amplitudes_dBm, label='Frequency Comb')
# plt.plot(comb_frequencies, comb_amplitudes_dBm, label='Original Frequency Domain', linestyle='--')
# plt.xlabel('Frequency (GHz)')
# plt.ylabel('Frequency Comb Amplitude')
# plt.title('Frequency Comb vs Frequency')
# plt.grid(True)
# plt.legend()
# plt.show()

# Build the on-site Hamiltonian terms
Hz_terms = []
Hx_terms = []

for i in range(N):
    # On-site sigma_z term with frequency shifts
    Hz_i = operator_at_site(sz, i, N)
    # Hz_terms.append([Hz_i, lambda t, args: omega_shifted(t, args) / 2 + args['dipole_moment'] * args['E_field']])
    Hz_terms.append([Hz_i, lambda t, args: omega_shifted(t, args) / 2 + args['dipole_moment'] * args['E_field']])

    # On-site sigma_x term with frequency comb
    Hx_i = operator_at_site(sx, i, N)
    Hx_terms.append([Hx_i, frequency_comb])  # Ensure frequency_comb uses omega_shifted
    # Hx_terms.append([Hx_i, Omega_R * signal_interpolator])#frequency_comb])  # Ensure frequency_comb uses omega_shifted


# Interaction terms
H_int_terms = []

# Heisenberg nearest-neighbor interactions
# for i in range(N - 1):
#     # sigma_x sigma_x coupling
#     Hxx = operator_at_site(sx, i, N) * operator_at_site(sx, i + 1, N)
#     H_int_terms.append([Hxx, lambda t, args: args['J']])

#     # sigma_y sigma_y coupling
#     Hyy = operator_at_site(sy, i, N) * operator_at_site(sy, i + 1, N)
#     H_int_terms.append([Hyy, lambda t, args: args['J']])

#     # sigma_z sigma_z coupling
#     Hzz = operator_at_site(sz, i, N) * operator_at_site(sz, i + 1, N)
#     H_int_terms.append([Hzz, lambda t, args: args['J']])

# Ising all-to-all interactions
total_sz = sum([operator_at_site(sz, i, N) for i in range(N)])
total_sy = sum([operator_at_site(sy, i, N) for i in range(N)])
total_sx = sum([operator_at_site(sx, i, N) for i in range(N)])

H_int_terms.append([total_sz * total_sz, lambda t, args: args['J'] / 2])
H_int_terms.append([total_sy * total_sy, lambda t, args: args['J'] / 2])
H_int_terms.append([total_sx * total_sx, lambda t, args: args['J'] / 2])

# Static terms (time-independent)
H_static = H_int_terms

# Time-dependent terms
H = H_static + Hz_terms + Hx_terms
print('Hamiltonian constructed.')

# Initial state: all qubits in ground state
psi_0 = tensor([basis(2, 0) for _ in range(N)])

# List of expectation operators (e.g., total magnetization along x)
expect_ops = [sum([operator_at_site(sx, i, N) for i in range(N)]), 
              sum([operator_at_site(sm, i, N) for i in range(N)]), 
              sum([operator_at_site(sm.dag(), i, N) for i in range(N)]),
              sum([operator_at_site(sz, i, N) for i in range(N)])]

# Number of expectation values and shifts
num_expectations = len(expect_ops)

# Create the subplots grid
fig, axs = plt.subplots(nrows=num_shifts, ncols=num_expectations, figsize=(num_expectations * 5, num_shifts * 4))

# Ensure axs is a 2D array
if num_expectations == 1:
    axs = np.array([axs]).T
if num_shifts == 1:
    axs = np.array([axs])
# plt.figure(figsize=(8, 6))

for idx, (shift_times, shift_magnitudes) in tqdm(enumerate(zip(shift_times, shift_magnitudes)), total=num_shifts, desc='Solving with shifts'):
    # Arguments for time-dependent functions
    args = {
        'omega_0': omega_0,
        'n_max': n_max,
        'Delta_omega': Delta_omega,
        'Omega_R': Omega_R,
        'shift_times': [shift_times],
        'shift_magnitudes': [shift_magnitudes],
        'E_field': E_field,
        'dipole_moment': dipole_moment,
        'J': J,
        # 'time_signal': time_signal
    }

    tqdm.write('Starting time evolution...')
    # Solve the time evolution
    result = sesolve(H, psi_0, t_eval, e_ops=expect_ops, args=args)
    tqdm.write('Time evolution completed.')


    # Extract expectation values
    expectation_sx_t = result.expect[0]  # Total magnetization along x
    expectation_sm_t = result.expect[1]  # Total magnetization along y
    expectation_sm_dag_t = result.expect[2]  # Total magnetization along y
    expectation_sz_t = result.expect[3]  # Total magnetization along z

    # FFT of the expectation value
    tqdm.write("Computing FFT of the expectation value...")
    N_t = len(t_eval)
    T = dt
    fft_sx = fft(expectation_sx_t)
    fft_sm = fft(expectation_sm_t)
    fft_sm_dag = fft(expectation_sm_dag_t)
    fft_sz = fft(expectation_sz_t)
    fft_freqs = fftfreq(N_t, T) + f_c  # Shift frequencies back to original range
    tqdm.write("FFT computed.")

    # Compute the absorption intensity
    absorption_intensity_sx = 2.0 / N_t * np.abs(fft_sx)
    absorption_intensity_sm = 2.0 / N_t * np.abs(fft_sm)
    absorption_intensity_sm_dag = 2.0 / N_t * np.abs(fft_sm_dag)
    absorption_intensity_sz = 2.0 / N_t * np.abs(fft_sz)

    # Plotting
    axs[idx, 0].plot(fft_freqs, absorption_intensity_sx, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 0].set_xlabel('Frequency (GHz)')
    axs[idx, 0].set_ylabel('Absorption Intensity')
    axs[idx, 0].set_title(f'Shift {idx + 1}: Total Sx')
    axs[idx, 0].grid(True)
    axs[idx, 0].legend()

    axs[idx, 1].plot(fft_freqs, absorption_intensity_sm, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 1].set_xlabel('Frequency (GHz)')
    axs[idx, 1].set_ylabel('Absorption Intensity')
    axs[idx, 1].set_title(f'Shift {idx + 1}: Total Sm')
    axs[idx, 1].grid(True)
    axs[idx, 1].legend()

    axs[idx, 2].plot(fft_freqs, absorption_intensity_sm_dag, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 2].set_xlabel('Frequency (GHz)')
    axs[idx, 2].set_ylabel('Absorption Intensity')
    axs[idx, 2].set_title(f'Shift {idx + 1}: Total Sm Dag')
    axs[idx, 2].grid(True)
    axs[idx, 2].legend()

    axs[idx, 3].plot(fft_freqs, absorption_intensity_sz, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 3].set_xlabel('Frequency (GHz)')
    axs[idx, 3].set_ylabel('Absorption Intensity')
    axs[idx, 3].set_title(f'Shift {idx + 1}: Total Sz')
    axs[idx, 3].grid(True)
    axs[idx, 3].legend()

plt.tight_layout()
plt.savefig('TLSCombNQubits.png')
plt.show()
