from TLS_Simulator import TLSSimulator, Hamiltonian, operator_at_site
import numpy as np
from qutip import tensor, basis, sigmam, qeye, sigmax, sigmaz, sigmay
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
# omega_d_list = np.array([args_template['omega_0']*0.97, args_template['omega_0'], args_template['omega_0']*1.03]) # For testing

# Operator for expectation value (excitation probability)
ground_state = tensor([basis(2, 0) for _ in range(args_template['N'])])
excited_state = tensor([basis(2, 1) for _ in range(args_template['N'])])
intertmediate_state = tensor([basis(2, 0) for _ in range(args_template['N'] - 1)] + [basis(2, 1)])

# Expectation operators
ex_ops = [
        #   sum(operator_at_site(sigmax(), i, args_template['N']) for i in range(args_template['N'])),
        #   sum(operator_at_site(sigmay(), i, args_template['N']) for i in range(args_template['N'])),
        #   sum(operator_at_site(sigmaz(), i, args_template['N']) for i in range(args_template['N'])),
          ground_state * ground_state.dag(),
          intertmediate_state * intertmediate_state.dag(),
          excited_state * excited_state.dag(),
          ]

# Prepare a list to store the results
expectation_values = [[] for _ in range(len(ex_ops))]  # Will be a 3D list: [expectation_value][frequency_index][time_index]
fourier_amplitudes = [[] for _ in range(len(ex_ops))]  # Will store the Fourier amplitudes if needed

# Loop over driving frequencies
for omega_d in tqdm(omega_d_list, desc='Sweeping over driving frequencies'):
    # Update the arguments for this iteration
    args = args_template.copy()
    args['omega_d'] = omega_d  # Add the current driving frequency to args
    args['driving_frequency'] = driving_frequency

    # Initialize the Hamiltonian
    hamiltonian = Hamiltonian(args=args)

    # Get eigenvalues and eigenvectors of the Hamiltonian
    # eigenvalues, eigenvectors = hamiltonian.eigenstates()

    # Initial state of the system
    psi_0 = tensor([basis(2, 0) for _ in range(args['N'])])

    # Initialize the simulator
    simulator = TLSSimulator(hamiltonian.get_H(), psi0=psi_0, tlist=tlist, e_ops=ex_ops, args=args)

    # Run the simulation
    results = simulator.run()

    # Sanity check: groud and excited state populations should sum to 1
    # assert np.allclose(results.expect[0] + results.expect[1], 1), f'Excited state and ground state sum up to {results.expect[0] + results.expect[1]} instead of 1'

    # Store the expectation values (taking the size of the complex numbers squared)
    for i in range(len(ex_ops)):
        expectation_values[i].append(results.expect[i]) 

    # simulator.plot_results(results, same_plot=True, titles=['Ground State Population', 'Intermediate State Population', 'Excited State Population'], x_label='Time (s)', y_label='Population', labels=['Ground State', 'Intermediate State', 'Excited State'])
    # simulator.plot_results(results, fourier_transform=True, titles=['Ground State Fourier Transform', 'Intermediate State Fourier Transform', 'Excited State Fourier Transform'], x_label='Frequency (Hz)', y_label='Fourier Amplitude', labels=['Ground State', 'Intermediate State', 'Excited State'])

    # Plot the results
    # Uncomment to plot the results for sanity check
    # plt.figure(figsize=(10, 6))
    # plt.plot(tlist, results.expect[0], label='Ground State Population')
    # plt.plot(tlist, results.expect[1], label='Intermediate State Population')
    # plt.plot(tlist, results.expect[2], label='Excited State Population')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Population')
    # plt.title(f'Population vs. Time for Driving Frequency {omega_d:.2e}')
    # plt.legend()
    # plt.show()

    for i in range(len(ex_ops)):
        results_fft = fftshift(fft(results.expect[i]))
        fourier_amplitudes[i].append(np.abs(results_fft))

# Optional: Compute and store Fourier amplitudes
time_spacing = tlist[1] - tlist[0]
freq_vector = fftshift(fftfreq(len(tlist), time_spacing))

titles = ['Ground State Population', 'Intermediate State Population', 'Excited State Population']

for i in range(len(expectation_values)):
    # Convert lists to arrays for easier plotting
    expectation_value = np.array(expectation_values[i])  # Shape: (num_frequencies, num_time_points)
    fourier_amplitude = np.array(fourier_amplitudes[i])  # Uncomment if computing Fourier transforms

    # Create the heatmap for expectation values
    normalized_omega_d_list = omega_d_list / args_template['omega_0']
    extent = [normalized_omega_d_list[0], normalized_omega_d_list[-1], tlist[0], tlist[-1]]

    plt.figure(figsize=(10, 6))
    plt.imshow(expectation_value.T, aspect='auto', extent=extent, origin='lower', cmap='viridis')
    plt.colorbar(label=f'{titles[i]}')
    plt.xlabel('Driving Frequency / TLS Base frequency')
    plt.ylabel('Time (s)')
    plt.title(f'{titles[i]} vs. Driving Frequency and Time')
    plt.show()

    # Create the heatmap for Fourier amplitudes
    # Uncomment if computing Fourier transforms
    extent_freq = [normalized_omega_d_list[0], normalized_omega_d_list[-1], freq_vector[0], freq_vector[-1]]
    plt.figure(figsize=(10, 6))
    plt.imshow(2.0 / len(tlist) * np.abs(fourier_amplitude.T), aspect='auto', extent=extent_freq, origin='lower', cmap='viridis')
    plt.colorbar(label=f'{titles[i]} Fourier Amplitude')
    plt.xlabel('Driving Frequency / TLS Base frequency')
    plt.ylabel('Fourier Frequency (Hz)')
    plt.title(f'{titles[i]} Fourier Transform vs. Driving Frequency and Fourier Frequency')
    plt.show()