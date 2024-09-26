print("Importing necessary libraries...")
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, basis, sesolve, sigmam
from scipy.fft import fft, fftfreq
from tqdm import tqdm
print("Imported necessary libraries")

# Number of TLS
N = 5  # Adjust N as needed

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
num_shifts = 3  # Number of random shifts
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

# Compute the frequency comb over t_eval
print('Computing frequency comb values for plotting...')
comb_values = np.array([frequency_comb(t, 
                                       {
                                        'n_max': n_max,
                                        'Delta_omega': Delta_omega,
                                        'omega_0': omega_0,
                                        'Omega_R': Omega_R
                                        }) for t in t_eval])

fft_comb = fft(comb_values)
fft_freqs_comb = fftfreq(len(t_eval), dt)

# Plot the frequency comb
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs_comb / 1e9, fft_comb, label='Frequency Comb')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Frequency Comb Amplitude')
plt.title('Frequency Comb vs Frequency')
plt.grid(True)
plt.show(block=False)

# Build the on-site Hamiltonian terms
Hz_terms = []
Hx_terms = []

for i in range(N):
    # On-site sigma_z term with frequency shifts
    Hz_i = operator_at_site(sz, i, N)
    Hz_terms.append([Hz_i, lambda t, args: omega_shifted(t, args) / 2 + args['dipole_moment'] * args['E_field']])

    # On-site sigma_x term with frequency comb
    Hx_i = operator_at_site(sx, i, N)
    Hx_terms.append([Hx_i, frequency_comb])  # Ensure frequency_comb uses omega_shifted

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

print('Hamiltonian constructed.')
# Time-dependent terms
H = H_static + Hz_terms + Hx_terms

# Initial state: all qubits in ground state
psi_0 = tensor([basis(2, 0) for _ in range(N)])

# List of expectation operators (e.g., total magnetization along x)
expect_ops = [sum([operator_at_site(sx, i, N) for i in range(N)]), 
              sum([operator_at_site(sm, i, N) for i in range(N)]), 
              sum([operator_at_site(sm.dag(), i, N) for i in range(N)])]

# Number of expectation values and shifts
num_expectations = len(expect_ops)

# Create the subplots grid
fig, axs = plt.subplots(nrows=num_shifts, ncols=num_expectations, figsize=(num_expectations * 5, num_shifts * 4))

# Ensure axs is a 2D array
if num_expectations == 1:
    axs = np.array([axs])
if num_shifts == 1:
    axs = np.array([axs]).T
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
        'J': J
    }

    # print('Starting time evolution...')
    # Solve the time evolution
    result = sesolve(H, psi_0, t_eval, e_ops=expect_ops, args=args)
    # print('Time evolution completed.')


    # Extract expectation values
    expectation_sx_t = result.expect[0]  # Total magnetization along x
    expectation_sm_t = result.expect[1]  # Total magnetization along y
    expectation_sm_dag_t = result.expect[2]  # Total magnetization along y

    # FFT of the expectation value
    # print("Computing FFT of the expectation value...")
    N_t = len(t_eval)
    T = dt
    fft_sx = fft(expectation_sx_t)[:N_t // 2] # Use only the positive frequencies
    fft_sm = fft(expectation_sm_t)[:N_t // 2] # Use only the positive frequencies
    fft_sm_dag = fft(expectation_sm_dag_t)[:N_t // 2] # Use only the positive frequencies
    fft_freqs = fftfreq(N_t, T)[:N_t // 2] # Use only the positive frequencies
    # print("FFT computed.")

    # Compute the absorption intensity
    absorption_intensity_sx = 2.0 / N_t * np.abs(fft_sx)
    absorption_intensity_sm = 2.0 / N_t * np.abs(fft_sm)
    absorption_intensity_sm_dag = 2.0 / N_t * np.abs(fft_sm_dag)

    # Plotting
    axs[idx, 0].plot(fft_freqs / 1e9, absorption_intensity_sx, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 0].set_xlabel('Frequency (GHz)')
    axs[idx, 0].set_ylabel('Absorption Intensity')
    axs[idx, 0].set_title(f'Shift {idx + 1}: Total Sx')
    axs[idx, 0].grid(True)
    axs[idx, 0].legend()

    axs[idx, 1].plot(fft_freqs / 1e9, absorption_intensity_sm, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 1].set_xlabel('Frequency (GHz)')
    axs[idx, 1].set_ylabel('Absorption Intensity')
    axs[idx, 1].set_title(f'Shift {idx + 1}: Total Sm')
    axs[idx, 1].grid(True)
    axs[idx, 1].legend()

    axs[idx, 2].plot(fft_freqs / 1e9, absorption_intensity_sm_dag, label=f"Shift at {shift_times:.2e} s\nMagnitude {shift_magnitudes:.2e} Hz")
    axs[idx, 2].set_xlabel('Frequency (GHz)')
    axs[idx, 2].set_ylabel('Absorption Intensity')
    axs[idx, 2].set_title(f'Shift {idx + 1}: Total Sm Dag')
    axs[idx, 2].grid(True)
    axs[idx, 2].legend()

plt.tight_layout()
plt.show()
