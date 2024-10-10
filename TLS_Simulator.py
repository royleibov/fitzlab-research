# print("Importing necessary libraries...")
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, sesolve, mesolve, sigmam, Qobj, coefficient#, basis
from scipy.fft import fft, fftfreq, fftshift#, ifftshift, ifft
# from tqdm import tqdm
# from scipy.interpolate import interp1d
from typing import Callable
# import math
# print("Imported necessary libraries")

class TLSSimulator:
    def __init__(self, H: Qobj, psi0: Qobj, tlist: np.array, e_ops: list, args: dict, c_ops: list = []):
        '''
        Simulates the time evolution of a TLS system.

        Parameters:
            H (Qobj): Hamiltonian of the system.
            psi0 (Qobj): Initial state of the system.
            tlist (array): List of times at which to evaluate the expectation values.
            e_ops (list): List of operators whose expectation values are to be evaluated.
            args (dict): Dictionary of parameters to pass to the Hamiltonian.

        Returns:
            None
        '''

        self.H = H
        self.psi0 = psi0
        self.tlist = tlist
        self.e_ops = e_ops
        self.args = args
        self.c_ops = c_ops

    def run(self):
        '''
        Simulates the time evolution of the system.

        Parameters:
            None

        Returns:
            results (list): List of expectation values of the operators at each time.
        '''
        if self.c_ops:
            results = mesolve(self.H, self.psi0, self.tlist, self.c_ops, self.e_ops, args=self.args, options={'nsteps': 5000})
        else:
            results = sesolve(self.H, self.psi0, self.tlist, self.e_ops, args=self.args)

        return results
    
    def plot_results(self, results: list, fourier_transform: bool = False, same_plot: bool = False, titles: list = None, x_label: str = 'Time', y_label: str = 'Expectation Value', labels: list = None):
        '''
        Plots the results of the simulation.

        Parameters:
            results (list): List of expectation values of the operators at each time.
            fourier_transform (bool): Whether to plot the Fourier transform of the results.
            same_plot (bool): Whether to plot all results on the same plot.
            titles (list): List of titles for each subplot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            labels (list): List of labels for the legend.

        Returns:
            None
        '''

        if fourier_transform:
            self.plot_fourier_transform(results, same_plot, titles, x_label, y_label, labels)
        else:
            if same_plot:
                plt.figure(figsize=(10, 6))
                for i in range(len(self.e_ops)):
                    label = labels[i] if labels else f'Operator {i}'
                    plt.plot(self.tlist, results.expect[i], linestyle='-', label=label)
                plt.title(titles[0] if titles else 'Expectation Values of Operators vs. Time')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                fig, axs = plt.subplots(len(self.e_ops), 1, figsize=(10, 6 * len(self.e_ops)))

                # Ensure axs is always iterable
                if len(self.e_ops) == 1:
                    axs = [axs]  # Wrap the single Axes object in a list

                for i in range(len(self.e_ops)):
                    label = labels[i] if labels else f'Operator {i}'
                    axs[i].plot(self.tlist, results.expect[i], linestyle='-', label=label)
                    axs[i].set_title(titles[i] if titles else f'Expectation Value of Operator {i} vs. Time')
                    axs[i].set_xlabel(x_label)
                    axs[i].set_ylabel(y_label)
                    axs[i].legend()
                    axs[i].grid(True)
                plt.tight_layout()
                plt.show()

    def plot_fourier_transform(self, results: list, same_plot: bool = False, titles: list = None, x_label: str = 'Frequency (Hz)', y_label: str = 'Amplitude', labels: list = None):
        '''
        Plots the Fourier transform of the results of the simulation.

        Parameters:
            results (list): List of expectation values of the operators at each time.
            same_plot (bool): Whether to plot all results on the same plot.
            titles (list): List of titles for each subplot.
            x_label (str): Label for the x-axis.
            y_label (str): Label for the y-axis.
            labels (list): List of labels for the legend.

        Returns:
            None
        '''

        if same_plot:
            plt.figure(figsize=(10, 6))
            # Calculate the Fourier transform of the results
            time_vector = self.tlist
            time_spacing = time_vector[1] - time_vector[0]
            freq_vector = fftfreq(len(time_vector), time_spacing)
            freq_vector = fftshift(freq_vector)

            for i in range(len(self.e_ops)):
                results_fft = fft(results.expect[i])
                results_fft = fftshift(results_fft)
                label = labels[i] if labels else f'Operator {i}'

                # Plot the Fourier transform
                plt.plot(freq_vector, np.abs(results_fft), linestyle='-', label=label)

            plt.title(titles[0] if titles else 'Fourier Transform of Expectation Values of Operators')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            fig, axs = plt.subplots(len(self.e_ops), 1, figsize=(10, 6 * len(self.e_ops)))

            # Ensure axs is always iterable
            if len(self.e_ops) == 1:
                axs = [axs]  # Wrap the single Axes object in a list
            
            for i in range(len(self.e_ops)):
                # Calculate the Fourier transform of the results
                time_vector = self.tlist
                time_spacing = time_vector[1] - time_vector[0]
                freq_vector = fftfreq(len(time_vector), time_spacing)
                freq_vector = fftshift(freq_vector)
                results_fft = fft(results.expect[i])
                results_fft = fftshift(results_fft)
                label = labels[i] if labels else f'Operator {i}'

                # Plot the Fourier transform
                axs[i].plot(freq_vector, np.abs(results_fft), linestyle='-', label=label)
                axs[i].set_title(titles[i] if titles else f'Fourier Transform of Expectation Value of Operator {i}')
                axs[i].set_xlabel(x_label)
                axs[i].set_ylabel(y_label)
                axs[i].legend()
                axs[i].grid(True)
            plt.tight_layout()
            plt.show()


class Hamiltonian:
    def __init__(self, args: dict):
        '''
        Initializes the Hamiltonian of the system.

        Parameters:
            args (dict): Dictionary of parameters to pass to the Hamiltonian.
            {
            N (int): Number of two-level systems.
            driving_frequency (float): Driving frequency of the external field.
            omega_0 (float): Energy level spacing.
            omega_R (float): Rabi frequency.
            J (float): Coupling strength between the two-level systems.
            E_field (float): Electric field strength.
            dipole_moment (float): Dipole moment of the system.
            }

        Returns:
            None
        '''
        N = args['N']
        driving_frequency = args['driving_frequency']
        J = args['J']

        I = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        sm = sigmam()
        
        # Build the on-site Hamiltonian terms
        Hz_terms = []
        Hx_terms = []

        for i in range(N):
            # On-site sigma_z term
            Hz_i = operator_at_site(sz, i, N)
            Hz_terms.append([Hz_i, lambda t, args: (args['omega_0'] / 2) + args['dipole_moment'] * args['E_field']])

            # On-site sigma_x term with driving frequency
            Hx_i = operator_at_site(sx, i, N)
            Hx_terms.append([Hx_i, driving_frequency])

        # Interaction terms
        H_int_terms = []

        if N > 1:
            # Ising all-to-all interactions
            total_sz = sum([operator_at_site(sz, i, N) for i in range(N)])
            total_sy = sum([operator_at_site(sy, i, N) for i in range(N)])
            total_sx = sum([operator_at_site(sx, i, N) for i in range(N)])
            total_I = sum([operator_at_site(I, i, N) for i in range(N)])

            H_int_terms.append([total_sz * total_sz - total_I, lambda t, args: args['J'] / 2])
            H_int_terms.append([total_sy * total_sy - total_I, lambda t, args: args['J'] / 2])
            H_int_terms.append([total_sx * total_sx - total_I, lambda t, args: args['J'] / 2])

        H_static = H_int_terms

        # Time-dependent terms
        H = H_static + Hz_terms + Hx_terms

        self.H = H

    def get_H(self):
        '''
        Returns the Hamiltonian of the system.

        Parameters:
            None

        Returns:
            H (Qobj): Hamiltonian of the system.
        '''
        return self.H
    
    def eigenstates(self):
        '''
        Returns the eigenvalues and eigenvectors of the Hamiltonian.

        Parameters:
            None

        Returns:
            eigenvalues (array): Eigenvalues of the Hamiltonian.
            eigenvectors (array): Eigenvectors of the Hamiltonian.
        '''
        H = Qobj(self.H)
        return H.eigenstates()
    
class RandomizedHamiltonian(Hamiltonian):
    def __init__(self, args: dict):
        N = args['N']
        driving_frequency = args['driving_frequency']
        omega_0_list = args['omega_0_list']
        J_matrix = args['J_matrix']
        phi_list = args['phi_list']

        I = qeye(2)
        sx = sigmax()
        sy = sigmay()
        sz = sigmaz()
        sm = sigmam()

        # Precompute operators at each site to avoid redundant computations
        sx_list = [operator_at_site(sx, i, N) for i in range(N)]
        sy_list = [operator_at_site(sy, i, N) for i in range(N)]
        sz_list = [operator_at_site(sz, i, N) for i in range(N)]
        I_list  = [operator_at_site(I,  i, N) for i in range(N)]

        # On-site Hamiltonian terms
        Hz_terms = []
        Hx_terms = []

        for i in range(N):
            # On-site sigma_z term with random omega_0
            Hz_i = sz_list[i]
            omega_0_i = omega_0_list[i]
            phi_i = phi_list[i]
            def Hz_i_func(t, args):
                return args['omega_0_i'] / 2 + args['dipole_moment'] * args['E_field'] * np.cos(args['phi_i'])
            Hz_i_coeff = coefficient(Hz_i_func, args={**args, 'omega_0_i': omega_0_i, 'phi_i': phi_i})
            Hz_terms.append([Hz_i, Hz_i_coeff])

            # On-site sigma_x term with driving frequency
            Hx_i = sx_list[i]
            Hx_terms.append([Hx_i, driving_frequency])

        # Interaction terms
        H_int_terms = []

        if N > 1:
            indices = np.triu_indices(N, k=1)

            for i, j in zip(indices[0], indices[1]):
                J_ij = J_matrix[i][j]

                # Bind J_ij to the lambda function
                def interaction_func(t, args):
                    return args['J_ij'] / 2
                interaction_coeff = coefficient(interaction_func, args={**args, 'J_ij': J_ij})
                H_int_terms.extend([
                    [sx_list[i] * sx_list[j], interaction_coeff],
                    [sy_list[i] * sy_list[j], interaction_coeff],
                    [sz_list[i] * sz_list[j], interaction_coeff]
                ])

        H_static = H_int_terms

        # Time-dependent terms
        self.H = H_static + Hz_terms + Hx_terms
    

def tensor_operator(op_list):
    return tensor(op_list)

def operator_at_site(op, site, N):
    I = qeye(2)
    op_list = [I] * N
    op_list[site] = op
    return tensor_operator(op_list)