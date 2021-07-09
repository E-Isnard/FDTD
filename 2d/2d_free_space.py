from FDTD import FDTD
import numpy as np

f = 1e9
eps = 8.85418782e-12
mu = 4*np.pi*1e-7
c = (eps*mu)**(-1/2)
l = c/f
delta = l/20
dt = 0.95*delta/c
L = 200*delta
d = 2
def source_func(t):
	pulse = np.sin(2*np.pi*f*t)
	return pulse
source_pos = (L/2,L/2)
L_slab = L
slab_pos = 0
epsr_func = lambda t: 1
T_max = 200*dt

fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func)
fdtd.run()
fdtd.animContour(-0,0.1,cmap="jet")
fdtd.animSurface(-0.2,0.2)