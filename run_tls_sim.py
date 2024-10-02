from TLS_Simulator import TLSSimulator, Hamiltonian, operator_at_site
import numpy as np
from qutip import tensor, basis, sigmam
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft, fftshift, fftfreq

# Define the parameters of the TLS simulation
args_template = {
    'N': 5,  # Number of TLSs
    'omega_0': 5e9,  # Base frequency of the TLS (Hz)
    'omega_R': 0.1e9,  # Rabi frequency (Hz)
    'J': 1e6,  # Coupling strength (Hz)
    'E_field': 0.1e6,  # Electric field strength (Hz)
    'dipole_moment': 4.0,  # Dipole moment of the TLS
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

# Define the range of driving frequencies to sweep over
omega_d_list = np.linspace(0.95 * args_template['omega_0'], 1.05 * args_template['omega_0'], 50)
# omega_d_list = np.array([args_template['omega_0']*0.9, args_template['omega_0'], args_template['omega_0']*1.1]) # For testing

# Prepare a list to store the results
expectation_values = []  # Will be a 2D list: [frequency_index][time_index]
fourier_amplitudes = []  # Will store the Fourier amplitudes if needed

# Operator for expectation value (excitation probability)
sm = sigmam()
proj_e = sm.dag() * sm  # Projector onto the excited state
proj_g = sm * sm.dag()  # Projector onto the ground state

# Loop over driving frequencies
for omega_d in tqdm(omega_d_list, desc='Sweeping over driving frequencies'):
    # Update the arguments for this iteration
    args = args_template.copy()
    args['omega_d'] = omega_d  # Add the current driving frequency to args
    args['driving_frequency'] = driving_frequency

    # Initialize the Hamiltonian
    hamiltonian = Hamiltonian(args=args)

    # Initial state of the system
    psi_0 = tensor([basis(2, 0) for _ in range(args['N'])])

    # Expectation operators
    ex_ops = [sum(operator_at_site(proj_e, i, args['N']) for i in range(args['N'])),
              sum(operator_at_site(proj_g, i, args['N']) for i in range(args['N']))]

    # Initialize the simulator
    simulator = TLSSimulator(hamiltonian.get_H(), psi0=psi_0, tlist=tlist, e_ops=ex_ops, args=args)

    # Run the simulation
    results = simulator.run()

    # Sanity check: groud and excited state populations should sum to 1
    assert np.allclose(results.expect[0] + results.expect[1], args['N'])

    # Store the expectation values (taking the size of the complex numbers squared)
    expectation_values.append(results.expect[0])  # Assuming one operator in e_ops

    # Plot the results
    # Uncomment to plot the results for sanity check
    # plt.figure(figsize=(10, 6))
    # plt.plot(tlist, results.expect[0], label='Excited State Population')
    # plt.plot(tlist, results.expect[1], label='Ground State Population')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Population')
    # plt.title(f'Population vs. Time for Driving Frequency {omega_d:.2e}')
    # plt.legend()
    # plt.show()

    # Optional: Compute and store Fourier amplitudes
    # time_spacing = tlist[1] - tlist[0]
    # results_fft = fftshift(fft(results.expect[0]))
    # freq_vector = fftshift(fftfreq(len(tlist), time_spacing))
    # fourier_amplitudes.append(np.abs(results_fft))

# Convert lists to arrays for easier plotting
expectation_values = np.array(expectation_values)  # Shape: (num_frequencies, num_time_points)
# fourier_amplitudes = np.array(fourier_amplitudes)  # Uncomment if computing Fourier transforms

# Create the heatmap for expectation values
normalized_omega_d_list = omega_d_list / args_template['omega_0']
extent = [normalized_omega_d_list[0], normalized_omega_d_list[-1], tlist[0], tlist[-1]]

plt.figure(figsize=(10, 6))
plt.imshow(expectation_values.T, aspect='auto', extent=extent, origin='lower', cmap='viridis')
plt.colorbar(label='Excited State Population')
plt.xlabel('Driving Frequency / TLS Base frequency')
plt.ylabel('Time (s)')
plt.title('Expectation Value vs. Driving Frequency and Time')
plt.show()

# Create the heatmap for Fourier amplitudes
# Uncomment if computing Fourier transforms
# plt.figure(figsize=(10, 6))
# plt.imshow(fourier_amplitudes.T, aspect='auto', extent=extent, origin='lower', cmap='viridis')
# plt.colorbar(label='Fourier Amplitude')
# plt.xlabel('Driving Frequency / TLS Base frequency')
# plt.ylabel('Frequency (Hz)')
# plt.title('Fourier Transform vs. Driving Frequency and Frequency')
# plt.show()