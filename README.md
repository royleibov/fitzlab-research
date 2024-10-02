# Coupled Two Level Systems (TLS) Simulation Library
In this repo you can find a library to help simulate N coupled quantum two level systems (TLS).
The hamiltonian is solved using QuTiP. Helper functions are provided to build a predefined Hamiltonian and to plot the expectation values.

## Requirements
After cloning the repo, please create a virtual environment:
```bash
python3 -m venv venv &&
source venv/bin/activate
```

Then download the requirements:
```bash
pip install -r requirements.txt
```

## Usage
To use the simulation, first import:
```python
from TLS_Simulator import TLSSimulator, Hamiltonian, operator_at_site
from qutip import basis, tensor, sigmam
import numpy as np
```

Next, you would want to build the Hamiltonian. You can build your own Hamiltonian, or you can use the predefined one:
$$H = \sum_i \left( \frac{\omega_{0}}{2} \vec{E} \cdot \vec{d} \sigma_z^{(i)} + f(t) \sigma_x^{(i)} \right) + \frac{J}{2} \left[ \left( \sum_i \sigma_x^{(i)} \right)^2 + \left( \sum_i \sigma_y^{(i)} \right)^2 + \left( \sum_i \sigma_z^{(i)} \right)^2 - 3 N I \right]$$

Where $\omega_{0}$ is the TLS base frequency, $E$ is the planar electric field, $d$ is the TLS dipole moment, $f(t)$ is a possibly time dependent driving field perpendicular to TLS plane, and $J$ is TLS coupling strength. The second term in the Hamiltonian (involving $J$) is a all-to-all Ising model coupling.

You would need to define an `args` dict before using the provided `Hamiltonian` class:
```python
args = {
    'N': 5,  # Number of TLSs
    'omega_0': 5e9,  # Base frequency of the TLS (Hz)
    'omega_R': 0.1e9,  # Rabi frequency (Hz)
    'J': 1e6,  # Coupling strength (Hz)
    'E_field': 0.1e6,  # Electric field strength (Hz)
    'dipole_moment': 4.0,  # Dipole moment of the TLS
}
```
As well a possibly time dependent driving field, $f(t)$, with the following function signature:
```python
driving_frequency(t, args)
```
where args is a dict.

Then define a time list, for example:
```python
tlist = np.arange(0, 1e-6, 1e-9)
```

And the initial state and list of expactation operators:
```python
# Initial state of the system
psi_0 = tensor([basis(2, 0) for _ in range(args['N'])])

# Expectation operators
sm = sigmam()
ex_ops = [sum(operator_at_site(sm.dag() * sm, i, args['N']) for i in range(args['N'])),
          sum(operator_at_site(sm * sm.dag(), i, args['N']) for i in range(args['N']))]
```
Note: in the above example I'm using a helper method `operator_at_site` to help build the hermitian.

Finally, initialize the simulator and run it:
```python
simulator = TLSSimulator(hamiltonian.get_H(), psi0=psi_0, tlist=tlist, e_ops=ex_ops, args=args)

results = simulator.run()
```
`simulator.run()` returns the results object from the QuTiP solver.

You can plot the results with the simulator's helper functions or further manipulate them and plot on you own:
```python
simulator.plot_results(results)
```
`simulator.plot_results` also has an optional flag `fourier_transform: bool = False` set to false on default that will preform a fourier transform on the results and plot the in the frequency domain.

## Change Log
v1.0
 * Can simulate N coupled TLSs with all-to-all coupling and a monochromatic driving field.
