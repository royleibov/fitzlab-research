from TLS_Simulator import TLSSimulator, RandomizedHamiltonian, operator_at_site
import numpy as np
from qutip import tensor, basis, sigmam, qeye, sigmax, sigmaz, sigmay
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, fftshift, fftfreq
from matplotlib import animation
from matplotlib.animation import PillowWriter
from joblib import Parallel, delayed, parallel_backend
import os  # Added for filesystem operations
print("All modules imported successfully!")

##### Define the parameters of the TLS simulation #####

args_template = {
    'N': 5,  # Number of TLSs
    'omega_0': 5e9,  # Base frequency of the TLS (Hz)
    'omega_R': 0.1e9,  # Rabi frequency (Hz)
    'J': 1e6,  # Coupling strength (Hz)
    'E_field': 0.1e6,  # Electric field strength (Hz)
    'dipole_moment': 4.0,  # Dipole moment of the TLS
    'gamma': 0.5  # Decay rate of the TLS
}

# Time parameters
t_span = (0, 1e-6)  # Time span for the simulation (1 microsecond)
dt = 1e-9           # Time step (1 ns)
tlist = np.arange(t_span[0], t_span[1], dt)

# Define the driving frequency function for this omega_d
def driving_frequency(t, args):
    strength = np.pi / 2
    omega_R = args['omega_R']
    omega_d = args['omega_d']
    return omega_R * strength * np.cos(omega_d * t)

def random_dipole_orientations(N):
    """
    Generates N dipole orientations based on random angles phi_i and computes Theta_ij.
    """
    phi_i = np.random.uniform(0, np.pi, N)
    theta_i = np.random.uniform(0, 2 * np.pi, N)
    sin_phi = np.sin(phi_i)
    v = np.stack((sin_phi * np.cos(theta_i), sin_phi * np.sin(theta_i), np.cos(phi_i)), axis=1)
    cos_theta = np.clip(np.einsum('ij,kj->ik', v, v), -1.0, 1.0)
    theta_ij = np.arccos(cos_theta)
    return phi_i, theta_ij

# Define the range of driving frequencies to sweep over
omega_d_list = np.linspace(0.8 * args_template['omega_0'], 1.2 * args_template['omega_0'], 60)

# Operator for expectation value (excitation probability)
ground_state = tensor([basis(2, 0) for _ in range(args_template['N'])])
excited_state = tensor([basis(2, 1) for _ in range(args_template['N'])])
intermediate_state = tensor([basis(2, 0) for _ in range(args_template['N'] - 1)] + [basis(2, 1)])

# Expectation operators
ex_ops = [
    sum(operator_at_site(sigmax(), i, args_template['N']) for i in range(args_template['N'])),
    sum(operator_at_site(sigmay(), i, args_template['N']) for i in range(args_template['N'])),
    sum(operator_at_site(sigmaz(), i, args_template['N']) for i in range(args_template['N'])),
]

c_ops = [operator_at_site(np.sqrt(args_template['gamma']) * sigmam(), i, args_template['N']) for i in range(args_template['N'])]

# Number of randomized realizations
num_realizations = 5

# Prepare nd arrays to store the results (we'll load data into these later)
expectation_values = None  # We'll initialize these after loading checkpoints
fourier_amplitudes = None

############################################################################################################

##### Run the randomized simulations in Parallel #####

def run_simulation(realization_idx, omega_d_list, args_template, tlist, ex_ops, c_ops, checkpoint_dir):
    """
    Run a single randomized simulation over the given omega_d_list.
    """
    # Check if this realization has already been computed
    checkpoint_file = os.path.join(checkpoint_dir, f'realization_{realization_idx}.npz')
    if os.path.exists(checkpoint_file):
        print(f"Realization {realization_idx} already completed. Skipping computation.")
        return None, None

    # Initialize the expectation values and Fourier amplitudes for this realization
    local_expectation_values = np.zeros((len(ex_ops), len(omega_d_list), len(tlist)))
    local_fourier_amplitudes = np.zeros((len(ex_ops), len(omega_d_list), len(tlist)))

    # Randomize hyperparameters
    N = args_template['N']
    omega_0_list = np.random.uniform(4e9, 6e9, N)
    phi_list, theta_ij = random_dipole_orientations(N)
    indices = np.triu_indices(N, k=1)
    theta_ij = theta_ij[indices]
    r_ij = np.random.uniform(0.5e-10, 1.5e-10, len(indices[0]))
    scaling_factor = 1e24
    J_ij_values = (1 - 3 * np.cos(theta_ij) ** 2) / (r_ij ** 3 * scaling_factor)
    J_matrix = np.zeros((N, N))
    J_matrix[indices] = J_ij_values
    J_matrix = J_matrix + J_matrix.T

    # Initial state of the system
    psi_0 = tensor([basis(2, 0) for _ in range(N)])

    # Loop over driving frequencies
    for wd_idx, omega_d in enumerate(tqdm(omega_d_list, desc=f'Job {realization_idx} Sweeping over driving frequencies', leave=False, position=1)):
        # Update arguments
        args = args_template.copy()
        args['omega_d'] = omega_d
        args['driving_frequency'] = driving_frequency
        args['omega_0_list'] = omega_0_list
        args['J_matrix'] = J_matrix
        args['phi_list'] = phi_list

        # Initialize the Hamiltonian
        hamiltonian = RandomizedHamiltonian(args=args)

        # Initialize the simulator
        simulator = TLSSimulator(hamiltonian.get_H(), psi0=psi_0, tlist=tlist, e_ops=ex_ops, c_ops=c_ops, args=args)

        # Run the simulation
        results = simulator.run()

        # Store the expectation values
        for i in range(len(ex_ops)):
            local_expectation_values[i, wd_idx, :] = results.expect[i]

        # Store the Fourier amplitudes
        for i in range(len(ex_ops)):
            results_fft = fftshift(fft(results.expect[i]))
            local_fourier_amplitudes[i, wd_idx, :] = np.abs(results_fft)

    # Save the results to a checkpoint file
    np.savez(checkpoint_file, expectation_values=local_expectation_values, fourier_amplitudes=local_fourier_amplitudes)
    print(f"Realization {realization_idx} completed and saved.")
    return local_expectation_values, local_fourier_amplitudes

# Create 'checkpoints' directory if it doesn't exist
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Determine which realizations have already been computed
completed_realizations = []
for idx in range(num_realizations):
    checkpoint_file = os.path.join(checkpoint_dir, f'realization_{idx}.npz')
    if os.path.exists(checkpoint_file):
        completed_realizations.append(idx)

# Indices of realizations that need to be computed
realizations_to_run = [idx for idx in range(num_realizations) if idx not in completed_realizations]

if len(realizations_to_run) == 0:
    print('All realizations have been computed.')
else:
    print(f'Computing realizations: {realizations_to_run}')

    # Run the simulations in parallel for realizations that need to be computed
    with parallel_backend('loky', n_jobs=-1):
        with tqdm(total=len(realizations_to_run), desc='Running parallel simulations', position=0) as pbar:
            # Wrap the function to update the progress bar
            results = Parallel()(
                delayed(run_simulation)(idx, omega_d_list, args_template, tlist, ex_ops, c_ops, checkpoint_dir) for idx in realizations_to_run
            )
            pbar.update(1)

# Save the results
np.savez('results.npz', expectation_values=expectation_values, fourier_amplitudes=fourier_amplitudes)

# Initialize the arrays
expectation_values = np.zeros((num_realizations, len(ex_ops), len(omega_d_list), len(tlist)))
fourier_amplitudes = np.zeros((num_realizations, len(ex_ops), len(omega_d_list), len(tlist)))

# Load the results from the checkpoint files
for realization_idx in range(num_realizations):
    checkpoint_file = os.path.join(checkpoint_dir, f'realization_{realization_idx}.npz')
    if os.path.exists(checkpoint_file):
        data = np.load(checkpoint_file)
        expectation_values[realization_idx] = data['expectation_values']
        fourier_amplitudes[realization_idx] = data['fourier_amplitudes']
    else:
        print(f'Results for realization {realization_idx} not found.')

############################################################################################################

##### Create animations of the expectation values #####

# Optional: Compute and store Fourier amplitudes
time_spacing = tlist[1] - tlist[0]
freq_vector = fftshift(fftfreq(len(tlist), time_spacing))

# Normalize omega_d_list if necessary
normalized_omega_d_list = omega_d_list / args_template['omega_0']

# Titles for the expectation values
titles = ['Expectation X', 'Expectation Y', 'Expectation Z']

# Create animations
for i in range(len(titles)):
    # Extract the data for the chosen expectation value
    expectation_data = expectation_values[:, i, :, :]
    fourier_data = fourier_amplitudes[:, i, :, :]

    # Define the extent for the heatmaps
    extent_time = [normalized_omega_d_list[0], normalized_omega_d_list[-1], tlist[0], tlist[-1]]
    extent_freq = [normalized_omega_d_list[0], normalized_omega_d_list[-1], freq_vector[0], freq_vector[-1]]

    # Time-Domain Heatmap Animation
    fig_time, ax_time = plt.subplots(figsize=(10, 6))
    img_time = ax_time.imshow(expectation_data[0].T, aspect='auto', extent=extent_time, origin='lower', cmap='viridis')
    cbar_time = fig_time.colorbar(img_time, ax=ax_time)
    cbar_time.set_label(f'{titles[i]}')
    ax_time.set_xlabel('Driving Frequency / TLS Base Frequency')
    ax_time.set_ylabel('Time (s)')
    ax_time.set_title(f'{titles[i]} - Realization 1')

    def animate_time(realization_idx):
        img_time.set_data(expectation_data[realization_idx].T)
        ax_time.set_title(f'{titles[i]} - Realization {realization_idx + 1}')
        return [img_time]

    ani_time = animation.FuncAnimation(fig_time, animate_time, frames=num_realizations, blit=True)
    writer = PillowWriter(fps=2)
    ani_time.save(f'heatmap_{titles[i]}_time_domain.gif', writer=writer)
    plt.close(fig_time)

    # Frequency-Domain Heatmap Animation
    fig_freq, ax_freq = plt.subplots(figsize=(10, 6))
    img_freq = ax_freq.imshow(fourier_data[0].T, aspect='auto', extent=extent_freq, origin='lower', cmap='plasma')
    cbar_freq = fig_freq.colorbar(img_freq, ax=ax_freq)
    cbar_freq.set_label(f'Fourier Amplitude of {titles[i]}')
    ax_freq.set_xlabel('Driving Frequency / TLS Base Frequency')
    ax_freq.set_ylabel('Frequency (Hz)')
    ax_freq.set_title(f'Fourier Amplitude of {titles[i]} - Realization 1')

    def animate_freq(realization_idx):
        img_freq.set_data(fourier_data[realization_idx].T)
        ax_freq.set_title(f'Fourier Amplitude of {titles[i]} - Realization {realization_idx + 1}')
        return [img_freq]

    ani_freq = animation.FuncAnimation(fig_freq, animate_freq, frames=num_realizations, blit=True)
    ani_freq.save(f'heatmap_{titles[i]}_frequency_domain.gif', writer=writer)
    plt.close(fig_freq)

print("Animations created successfully!")
