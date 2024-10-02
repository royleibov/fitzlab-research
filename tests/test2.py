# -*- coding: utf-8 -*-
# Time:2020/8/21 15:38
# author: Junyong  
from qutip import *
from pylab import *
c = 299792458.0
tlist = linspace(0, 50e-12, 1000)

g = basis(2, 0)
e = basis(2, 1)

Omega_0 = 2*pi*c/(800e-9)   # atom transition wavelength 800nm
Omega   = 2*pi*c/(800e-9)   # laser field wavelength 800nm
delta   = Omega-Omega_0

area, FWHM, t_center = [pi, 5e-12, 20e-12]
w = FWHM/(2*sqrt(log(2)))


def rabi_frequency(t, args):    # Gaussian profile
    return 0.5 * area / (w * sqrt(pi)) * exp(-((t - t_center) / w) ** 2)


H0 = delta * e * e.dag()
H  = [H0, [e*g.dag()+g*e.dag(), rabi_frequency]]
c_ops = []
e_ops = [e * e.dag(), g * g.dag()]
psi0 = g
result = mesolve(H, psi0, tlist, c_ops, e_ops, progress_bar=True, options=Options(max_step=FWHM / 4))
ocp_X, ocp_G = [result.expect[0], result.expect[1]]

# plot
tlist = tlist*1e12
fig,(ax1,ax2,ax3) = subplots(3, 1, sharex=True)
ax1.plot(tlist, result.expect[0], label='|X>', lw=2, color='orange')
ax1.legend(loc=1)
ax1.grid(True)

ax2.plot(tlist, result.expect[1], label='|G>', lw=2, color='black')
ax2.set_ylabel(r'Population', size=15)
ax2.legend(loc=1)
ax2.grid(True)

ax3.fill_between(tlist, rabi_frequency(tlist,{}), label=r'Pulse', facecolor='red', alpha=0.4)
ax3.set_xlabel('Time [ps]', size=15)
ax3.legend(loc=1)
plt.figtext(0.5, 0.94, r'$ FWHM=%.1f ps, center=%.1f ps, area=%.1f pi $'
            % (FWHM*1e12, t_center*1e12, area/pi), ha='center')
show()