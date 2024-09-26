import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# Parameters
omega_0 = 5e9        # Base frequency of the defect (5 GHz)
Omega_R = 0.1e9      # Rabi frequency (100 MHz)
n_max = 100          # Number of frequency components in the comb
Delta_omega = 10e6   # Comb spacing (10 MHz)
t_span = (0, 1e-6)   # Time span for the simulation (1 microsecond)
dt = 1e-9            # Time step (1 ns)
t_eval = np.arange(t_span[0], t_span[1], dt)

# Static electric field parameters
E_field = 1e5        # Static electric field strength (V/m)
dipole_moment = 4.0  # Dipole moment of the TLS (C·m)

# Define Pauli matrices using QuTiP
sigma_x = sigmax()
sigma_z = sigmaz()

# Initial state (ground state)
psi_0 = basis(2, 0)

# Generate random frequency shifts
print('Generating random frequency shifts...')
num_shifts = 4                  # Number of random shifts
max_shift = 1e6                 # Maximum shift magnitude (1 MHz)
shift_times = np.sort(np.random.choice(t_eval, num_shifts, replace=False))
shift_magnitudes = np.random.uniform(-max_shift, max_shift, num_shifts)
print('Done!')

# Plot the random shifts over time
plt.figure(figsize=(8, 6))
shift_values = np.zeros_like(t_eval)
for shift_time, shift_magnitude in zip(shift_times, shift_magnitudes):
    shift_values[t_eval >= shift_time] = shift_magnitude
plt.step(t_eval, shift_values, where='post', label='Random frequency shifts', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Frequency shift (Hz)')
plt.title('Random Frequency Shifts vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Define functions for the time-dependent Hamiltonian
def omega_shifted(t, args):
    omega_0 = args['omega_0']
    shift_times = args['shift_times']
    shift_magnitudes = args['shift_magnitudes']
    idx = np.searchsorted(shift_times, t, side='right') - 1
    shift_applied = shift_magnitudes[idx] if idx >= 0 else 0.0
    return omega_0 + shift_applied

def frequency_comb(t, args):
    omega_shift = omega_shifted(t, args)
    Delta_omega = args['Delta_omega']
    n_max = args['n_max']
    delta_omega_t_over_2 = Delta_omega * t / 2
    numerator = np.sin((2 * n_max + 1) * delta_omega_t_over_2)
    denominator = np.sin(delta_omega_t_over_2)
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.isclose(denominator % (2 * np.pi), 0),
            np.cos(omega_shift * t) * (2 * n_max + 1),
            np.cos(omega_shift * t) * numerator / denominator
        )
    return result

def Hz(t, args):
    omega_shift = omega_shifted(t, args)
    E_field = args['E_field']
    dipole_moment = args['dipole_moment']
    return omega_shift / 2 + dipole_moment * E_field

def Hx(t, args):
    Omega_R = args['Omega_R']
    return Omega_R * frequency_comb(t, args)

# Define the time-dependent Hamiltonian in QuTiP format
H = [[sigma_z, Hz], [sigma_x, Hx]]

# Initialize a new figure for absorption spectra
plt.figure(figsize=(10, 8))

# Loop over each random shift and compute the absorption spectrum
for idx, (shift_time, shift_magnitude) in tqdm(enumerate(zip(shift_times, shift_magnitudes)),
                                               total=num_shifts, desc='Computing absorption spectra'):
    # Update args for the current shift
    args = {
        'omega_0': omega_0,
        'shift_times': [shift_time],
        'shift_magnitudes': [shift_magnitude],
        'E_field': E_field,
        'dipole_moment': dipole_moment,
        'Delta_omega': Delta_omega,
        'n_max': n_max,
        'Omega_R': Omega_R
    }
    
    # Solve the time evolution using QuTiP's sesolve
    result = sesolve(H, psi_0, t_eval, e_ops=[sigma_x], args=args)
    
    # Extract the expectation value of sigma_x over time
    expectation_sigma_x_t_shift = result.expect[0]
    
    # Compute the FFT of the expectation value
    N = len(t_eval)
    T = dt
    fft_sigma_x = fft(expectation_sigma_x_t_shift)
    fft_freqs = fftfreq(N, T)[:N // 2]
    
    # Plot the absorption spectrum
    plt.subplot(2, 2, idx + 1)
    plt.plot(fft_freqs / 1e9, 2.0 / N * np.abs(fft_sigma_x[:N // 2]),
             label=f'Shift at {shift_time:.2e} s\nMagnitude {shift_magnitude:.2e} Hz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Absorption Intensity')
    plt.title(f'Absorption Spectrum for Shift {idx + 1}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
