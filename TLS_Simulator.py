print("Importing necessary libraries...")
import numpy as np
import matplotlib.pyplot as plt
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, sesolve, sigmam, Qobj#, basis
from scipy.fft import fft, fftfreq, fftshift#, ifftshift, ifft
# from tqdm import tqdm
# from scipy.interpolate import interp1d
from typing import Callable
# import math
print("Imported necessary libraries")

class TLSSimulator:
    def __init__(self, H: Qobj, psi0: Qobj, tlist: np.array, e_ops: list, args: dict):
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

    def run(self):
        '''
        Simulates the time evolution of the system.

        Parameters:
            None

        Returns:
            results (list): List of expectation values of the operators at each time.
        '''

        results = sesolve(self.H, self.psi0, self.tlist, self.e_ops, args=self.args)
        return results
    
    def plot_results(self, results: list, fourier_transform: bool = False):
        '''
        Plots the results of the simulation.

        Parameters:
            results (list): List of expectation values of the operators at each time.
            fourier_transform (bool): Whether to plot the Fourier transform of the results.

        Returns:
            None
        '''

        if fourier_transform:
            self.plot_fourier_transform(results)
        else:
            fig, axs = plt.subplots(len(self.e_ops), 1, figsize=(10, 6 * len(self.e_ops)))

            # Ensure axs is always iterable
            if len(self.e_ops) == 1:
                axs = [axs]  # Wrap the single Axes object in a list

            for i in range(len(self.e_ops)):
                axs[i].plot(self.tlist, results.expect[i], linestyle='-', label=f'Operator {i}')
                axs[i].set_title(f'Expectation Value of Operator {i} vs. Time')
                axs[i].set_xlabel('Time')
                axs[i].set_ylabel('Expectation Value')
                axs[i].legend()
                axs[i].grid(True)
            plt.tight_layout()
            plt.show()

    def plot_fourier_transform(self, results: list):
        '''
        Plots the Fourier transform of the results of the simulation.

        Parameters:
            results (list): List of expectation values of the operators at each time.

        Returns:
            None
        '''

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

            # Plot the Fourier transform
            axs[i].plot(freq_vector, np.abs(results_fft), linestyle='-')
            axs[i].set_title(f'Fourier Transform of Expectation Value of Operator {i}')
            axs[i].set_xlabel('Frequency (Hz)')
            axs[i].set_ylabel('Amplitude')
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
            Hz_terms.append([Hz_i, lambda t, args: args['omega_0'] / 2 + args['dipole_moment'] * args['E_field']])

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
    

def tensor_operator(op_list):
    return tensor(op_list)

def operator_at_site(op, site, N):
    I = qeye(2)
    op_list = [I] * N
    op_list[site] = op
    return tensor_operator(op_list)