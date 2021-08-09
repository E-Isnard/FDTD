from FDTD import FDTD,progress
import numpy as np
import matplotlib.pyplot as plt
d=1
L=600e-3
T_max = 10e-9
delta = 1.5e-3

source_pos = 10*delta
ws = 1e10*2*np.pi
source_func = lambda t: np.sin(ws*t)

wm = 1e9*2*np.pi
b = 0.5
epsr_0 = 3
epsr_func = lambda t: epsr_0*(1+b*np.sin(wm*t))

L_slab = 93e-3
slab_pos = L/2-L_slab


fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func,boundary_condition="Mur")
fdtd.run()
fdtd.anim1d(-3,3)
fdtd.plotE1d(slab_pos+L_slab+2*delta)

energy = fdtd.energy()
t = np.linspace(0,fdtd.T_max,fdtd.nt)[1:]
plt.plot(t,energy)
plt.show()