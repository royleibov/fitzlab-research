from qutip import *
from scipy import *
import numpy as np
# from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Simulate data
np.random.seed(0)
frequencies = np.linspace(0, 10, 100)
Gains = [1, 10, 35]  # Different gains as per the paper
colors = ['cyan', 'green', 'black']

# Noise without squeezing (pump off)
noise_off = np.random.normal(0, 1, frequencies.shape)

# Function to simulate squeezed noise
def squeezed_noisen(gain):
    return noise_off / np.sqrt(gain)

def squeezed_noisea(gain):
    return noise_off * np.sqrt(gain)

# Plot the results
fig, (ax1, ax2) = plt.subplots(2,1)
tn1=tn2=[]
for gain, color in zip(Gains, colors):
    noise_on1 = squeezed_noisen(gain)
    noise_on2 = squeezed_noisea(gain)
    tn1.append(noise_on1)
    tn2.append(noise_on2)
    ax1.plot(frequencies, noise_on1, label=f'Pump on, Gain={gain} dB', color=color)
    ax2.plot(frequencies, noise_on2, label=f'Pump on, Gain={gain} dB', color=color)
ax1.plot(frequencies, noise_off, label='Pump off', color='blue', linestyle='dashed')
ax2.plot(frequencies, noise_off, label='Pump off', color='blue', linestyle='dashed')
# Adding titles and labels
ax1.set_title('Quadrature 1 Noise')
ax1.set_xlabel('Frequency (MHz)')
ax1.set_ylabel('Noise Power (a.u.)')
ax1.set_ylim(-3,3)
ax1.legend()
ax1.grid(True)

ax2.set_title('Quadrature 2 Noise')
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('Noise Power (a.u.)')
ax2.set_ylim(-15,15)
ax2.legend()
ax2.grid(True)

# Show the plot
plt.show()

####################################
tntotal=[]

