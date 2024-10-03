from TLS_Simulator import TLSSimulator, Hamiltonian, operator_at_site
import numpy as np
from qutip import tensor, basis, sigmax
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, fftshift, fftfreq
import matplotlib.animation as animation

# Parameters
args_template = {
    'N': 5,
    'omega_0': 5e9,       # Base frequency of the TLS (Hz)
    'omega_R': 0.1e9,     # Rabi frequency (Hz)
    'J': 1e6,             # Coupling strength (Hz)
    'dipole_moment': 4.0, # Dipole moment
    'E_field': 0.0,       # This will be swept over
}

# Time parameters
t_span = (0, 1e-6)  # Total simulation time
dt = 1e-9           # Time step
tlist = np.arange(t_span[0], t_span[1], dt)

# Define driving frequency function
def driving_frequency(t, args):
    strength = np.pi / 2
    omega_R = args['omega_R']
    omega_d = args['omega_d']
    return omega_R * strength * np.cos(omega_d * t)

# Electric field sweep (animation frames)
E_field_list = np.linspace(200.0, 500000000.0, 50)  # Adjust as needed

# Driving frequency sweep (X-axis)
omega_d_list = np.linspace(0.95 * args_template['omega_0'], 1.05 * args_template['omega_0'], 50)

# Operators for expectation values (sigma_x)
ex_ops = [sum(operator_at_site(sigmax(), i, args_template['N']) for i in range(args_template['N']))]

# Frequency vector for Fourier transform
time_spacing = tlist[1] - tlist[0]
freq_vector = fftshift(fftfreq(len(tlist), time_spacing))

# Initialize array to store Fourier amplitudes
num_E_fields = len(E_field_list)
num_omega_ds = len(omega_d_list)
num_freqs = len(tlist)
fourier_amplitudes = np.zeros((num_E_fields, num_omega_ds, num_freqs))

for idx_E, E_field in enumerate(tqdm(E_field_list, desc='Sweeping over E_field')):
    for idx_wd, omega_d in enumerate(tqdm(omega_d_list, desc='Sweeping over omega_d', leave=False)):
        args = args_template.copy()
        args['E_field'] = E_field
        args['omega_d'] = omega_d
        args['driving_frequency'] = driving_frequency

        # Initialize Hamiltonian
        hamiltonian = Hamiltonian(args=args)

        # Initial state (ground state)
        psi_0 = tensor([basis(2, 0) for _ in range(args['N'])])

        # Initialize the simulator
        simulator = TLSSimulator(hamiltonian.get_H(), psi0=psi_0, tlist=tlist, e_ops=ex_ops, args=args)

        # Run the simulation
        results = simulator.run()

        # Get sigma_x expectation value
        expectation_value = results.expect[0]

        # Compute Fourier transform
        results_fft = fftshift(fft(expectation_value))

        # Store Fourier amplitude
        fourier_amplitudes[idx_E, idx_wd, :] = np.abs(results_fft)

# Normalize omega_d for X-axis
normalized_omega_d_list = omega_d_list / args_template['omega_0']

# Compute the extent for imshow
extent = [normalized_omega_d_list[0], normalized_omega_d_list[-1], freq_vector[len(freq_vector)//2], freq_vector[-1]]

# Global min and max for color scaling
vmin = np.min(fourier_amplitudes[:, :, len(freq_vector)//2:])
vmax = np.max(fourier_amplitudes[:, :, len(freq_vector)//2:])

# Set up the figure
fig, ax = plt.subplots()

# Initialize the image
img = ax.imshow(fourier_amplitudes[0, :, len(freq_vector)//2:].T, aspect='auto', extent=extent, origin='lower',
                cmap='viridis', vmin=vmin, vmax=vmax)
ax.set_xlabel('Driving Frequency / Base TLS Frequency')
ax.set_ylabel('Fourier Frequency (Hz)')
ax.set_title(f'Fourier Transform of SigmaX Expectation\nE_field = {E_field_list[0]:.2e} V/m')
cbar = fig.colorbar(img, ax=ax)
cbar.set_label('Fourier Amplitude')

def animate(idx_E):
    img.set_data(fourier_amplitudes[idx_E, :, len(freq_vector)//2:].T)
    ax.set_title(f'Fourier Transform of SigmaX Expectation\nE_field = {E_field_list[idx_E]:.2e} V/m')
    return [img]

ani = animation.FuncAnimation(fig, animate, frames=num_E_fields, blit=True)

# Optionally save the animation
from matplotlib.animation import PillowWriter

# Set up the writer
writer = PillowWriter(fps=5)

# Save the animation as a GIF
ani.save('fourier_transform_animation.gif', writer=writer)

# plt.show()
