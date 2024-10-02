import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from tqdm import tqdm

# Define Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Parameters
omega_0 = 5e9  # Base frequency of the defect (5 GHz)
Omega_R = 0.1e9  # Rabi frequency (100 MHz, field strength)
n_max = 100       # Number of frequency components in the comb
Delta_omega = 10e6  # Comb spacing (10 MHz)
t_span = (0, 1e-6)  # Time span for the simulation (1 microsecond)
dt = 1e-9  # Time step (1 ns)
t_eval = np.arange(t_span[0], t_span[1], dt)

# Static electric field parameters
E_field = 100000.0  # Static electric field strength (V/m)
dipole_moment = 4.0  # Dipole moment of the TLS (in C·m, or alternatively, GHz·m)

# Optimized frequency comb function using analytical expression
def frequency_comb(t, omega_0, Delta_omega, n_max):
    delta_omega_t_over_2 = Delta_omega * t / 2
    numerator = np.sin((2 * n_max + 1) * delta_omega_t_over_2)
    denominator = np.sin(delta_omega_t_over_2)
    # Handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            np.isclose(denominator, 0),
            np.cos(omega_0 * t) * (2 * n_max + 1),
            np.cos(omega_0 * t) * numerator / denominator
        )
    return result

# Optimized function to apply random shifts
def apply_random_shift(t, shift_times, shift_magnitudes, omega_0):
    idx = np.searchsorted(shift_times, t, side='right') - 1
    shift_applied = shift_magnitudes[idx] if idx >= 0 else 0.0
    return omega_0 + shift_applied

# Define the Hamiltonian with random frequency shifts and static electric field
def H(t, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    if shift_times is not None and shift_magnitudes is not None:
        omega_shifted = apply_random_shift(t, shift_times, shift_magnitudes, omega_0)
    else:
        omega_shifted = omega_0  # No shift case
    comb_field = Omega_R * frequency_comb(t, omega_shifted, Delta_omega, n_max)

    # Add the interaction with the static electric field in the σ_z term
    electric_field_interaction = dipole_moment * E_field

    return (omega_shifted / 2 + electric_field_interaction) * sigma_z + comb_field * sigma_x

# Define the time-dependent Schrödinger equation
def schrodinger_eq(t, psi, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    psi = psi.reshape((2, 1))  # Convert to column vector
    H_t = H(t, Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times, shift_magnitudes)  # Time-dependent Hamiltonian
    dpsi_dt = -1j * np.dot(H_t, psi)  # Schrödinger equation
    return dpsi_dt.flatten()  # Flatten back to 1D for solver

# Initial state (ground state)
psi_0 = np.array([1, 0], dtype=complex)

# Optimized solver function
def solve_with_shifts(t_eval, psi_0, Omega_R, omega_0, Delta_omega, n_max,
                      E_field, dipole_moment, shift_times=None, shift_magnitudes=None):
    t_span = (t_eval[0], t_eval[-1])
    sol = solve_ivp(
        schrodinger_eq, t_span, psi_0, t_eval=t_eval,
        args=(Omega_R, omega_0, Delta_omega, n_max, E_field, dipole_moment, shift_times, shift_magnitudes),
        rtol=1e-9, atol=1e-9
    )
    psi_t = sol.y.T  # Transpose to get psi_t[n_time_steps, 2]
    # Renormalize psi_t
    norms = np.linalg.norm(psi_t, axis=1, keepdims=True)
    psi_t /= norms
    return psi_t

# Vectorized computation of the expectation value of sigma_x
def expectation_value_sigma_x(psi_t, sigma_x):
    return np.real(np.einsum('bi,ij,bj->b', np.conj(psi_t), sigma_x, psi_t))

# Generate random frequency shifts
print('Generating random frequency shifts...')
num_shifts = 4  # Number of random shifts
max_shift = 1e6  # Maximum shift magnitude (1 MHz)
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

# Initialize a new figure for absorption spectra
plt.figure(figsize=(10, 8))

# Plot the absorption spectra in a 2x2 grid
for idx, (shift_time, shift_magnitude) in tqdm(enumerate(zip(shift_times, shift_magnitudes)),
                                               total=num_shifts, desc='Computing absorption spectra'):
    # Perform the simulation with a specific shift
    psi_t_shift = solve_with_shifts(
        t_eval, psi_0, Omega_R, omega_0, Delta_omega, n_max,
        E_field, dipole_moment, [shift_time], [shift_magnitude]
    )

    # Compute the expectation value of sigma_x over time for this shift
    expectation_sigma_x_t_shift = expectation_value_sigma_x(psi_t_shift, sigma_x)

    # FFT of the expectation value of sigma_x
    N = len(t_eval)
    T = dt  # Time step (1 ns)
    fft_sigma_x = fft(expectation_sigma_x_t_shift)
    fft_freqs = fftfreq(N, T)[:N // 2]  # Only take the positive frequencies

    # Create subplots in a 2x2 grid
    plt.subplot(2, 2, idx + 1)  # 2x2 grid, place the plot in the next available spot
    plt.plot(fft_freqs / 1e9, 2.0 / N * np.abs(fft_sigma_x[:N // 2]),
             label=f'Shift at {shift_time:.2e} s\nMagnitude {shift_magnitude:.2e} Hz')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Absorption Intensity')
    plt.title(f'Absorption Spectrum for Shift {idx + 1}')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
