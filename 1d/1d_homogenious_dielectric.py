from fdtd import FDTD
import numpy as np

L=300e-3
delta = 1e-3
T_max=6e-9
d=1
source_func = lambda t: np.exp(-((t-T_max/20)*1e10)**2)
source_pos = L/10
L_slab = L/4
slab_pos = L/2
epsr_func = lambda t : 4

fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func)
fdtd.run()
fdtd.anim1d(-1,1)