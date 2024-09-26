print("Importing necessary libraries...")
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, sesolve
from scipy.fft import fft, fftfreq
from tqdm import tqdm
print("Imported necessary libraries")

# Number of TLS
N = 10  # Adjust N as needed

# System parameters
omega_0 = 5e9        # Base frequency of the TLS (5 GHz)
Omega_R = 0.1e9      # Rabi frequency (100 MHz)
n_max = 100          # Number of frequency components in the comb
Delta_omega = 10e6   # Comb spacing (10 MHz)
t_span = (0, 1e-6)   # Time span for the simulation (1 microsecond)
dt = 1e-9            # Time step (1 ns)
t_eval = np.arange(t_span[0], t_span[1], dt)

# Coupling strength between TLS
J = 1e6  # Coupling strength (1 MHz)

# Static electric field parameters
E_field = 100000.0  # Static electric field strength (V/m)
dipole_moment = 4.0  # Dipole moment of the TLS (in C·m, or alternatively, GHz·m)

# Generate random frequency shifts
print('Generating random frequency shifts...')
num_shifts = 4  # Number of random shifts
max_shift = 1e6  # Maximum shift magnitude (1 MHz)
shift_times = np.sort(np.random.choice(t_eval, num_shifts, replace=False))
shift_magnitudes = np.random.uniform(-max_shift, max_shift, num_shifts)
print('Done!')

# Plot the random frequency shifts
plt.figure()
plt.plot(t_eval, np.zeros_like(t_eval), 'k--', label='Base frequency')
plt.step(shift_times, shift_magnitudes, color='red', label='Random shifts')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Shift (Hz)')
plt.title('Random Frequency Shifts')
plt.legend()
plt.grid(True)
plt.show(block=False)

# Identity and Pauli operators
I = qeye(2)
sx = sigmax()
sy = sigmay()
sz = sigmaz()

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
    shift_magnitudes = args['shift_magnitudes']
    idx = np.searchsorted(shift_times, t, side='right') - 1
    if idx >= 0:
        shift_applied = shift_magnitudes[idx]
    else:
        shift_applied = 0.0
    return omega_0 + shift_applied

# Frequency comb function
def frequency_comb(t, args):
    n_max = args['n_max']
    Delta_omega = args['Delta_omega']
    omega_shift = omega_shifted(t, args)  # Use shifted omega
    Omega_R = args['Omega_R']
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
    return Omega_R * result

# Build the on-site Hamiltonian terms
Hz_terms = []
Hx_terms = []

for i in range(N):
    # On-site sigma_z term with frequency shifts
    Hz_i = operator_at_site(sz, i, N)
    Hz_terms.append([Hz_i, lambda t, args: omega_shifted(t, args) / 2, lambda t, args: args['dipole_moment'] * args['E_field']])

    # On-site sigma_x term with frequency comb
    Hx_i = operator_at_site(sx, i, N)
    Hx_terms.append([Hx_i, frequency_comb])  # Ensure frequency_comb uses omega_shifted

# Interaction terms
H_int_terms = []

for i in range(N - 1):
    # sigma_x sigma_x coupling
    Hxx = J * operator_at_site(sx, i, N) * operator_at_site(sx, i + 1, N)
    H_int_terms.append(Hxx)

    # sigma_y sigma_y coupling
    Hyy = J * operator_at_site(sy, i, N) * operator_at_site(sy, i + 1, N)
    H_int_terms.append(Hyy)

    # sigma_z sigma_z coupling
    Hzz = J * operator_at_site(sz, i, N) * operator_at_site(sz, i + 1, N)
    H_int_terms.append(Hzz)

# Static terms (time-independent)
H_static = sum(H_int_terms)

print('Hamiltonian constructed.')
# Time-dependent terms
H = [H_static] + Hz_terms + Hx_terms

# Initial state: all qubits in ground state
psi_0 = tensor([basis(2, 0) for _ in range(N)])

# List of expectation operators (e.g., total magnetization along x)
expect_ops = [sum([operator_at_site(sx, i, N) for i in range(N)])]

plt.figure(figsize=(8, 6))

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
        'dipole_moment': dipole_moment
    }

    # print('Starting time evolution...')
    # Solve the time evolution
    result = sesolve(H, psi_0, t_eval, e_ops=expect_ops, args=args)
    # print('Time evolution completed.')


    # Extract expectation values
    expectation_sx_t = result.expect[0]  # Total magnetization along x

    # FFT of the expectation value
    # print("Computing FFT of the expectation value...")
    N_t = len(t_eval)
    T = dt
    fft_sx = fft(expectation_sx_t)[:N_t // 2] # Use only the positive frequencies
    fft_freqs = fftfreq(N_t, T)[:N_t // 2] # Use only the positive frequencies
    # print("FFT computed.")

    # Compute the absorption intensity
    absorption_intensity = 2.0 / N_t * np.abs(fft_sx)

    # Plot the absorption spectrum
    plt.subplot(2, 2, idx + 1)
    plt.plot(fft_freqs / 1e9, absorption_intensity, label="Shift at {:.2e} s\nMagnitude {:.2e} Hz".format(shift_times, shift_magnitudes))
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Absorption Intensity')
    plt.title('Absorption Spectrum for Shift {}'.format(idx + 1))
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
