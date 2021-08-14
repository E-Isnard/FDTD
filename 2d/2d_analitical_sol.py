from fdtd import FDTD
import numpy as np
import matplotlib.pyplot as plt
delta = 0.02
L = 1
d = 2
T_max = 14e-9
source_func = lambda t: 0
source_pos = (L/2,L/2)
L_slab = L
slab_pos = 0
epsr_func = lambda t: 1
f = 1e9
E0_func = lambda X,Y : np.sin(np.pi*X)*np.sin(np.pi*Y)
fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func,E0_func = E0_func,boundary_condition="PEC")
fdtd.run()

x = np.linspace(0,L,fdtd.n_space)
y = np.linspace(0,L,fdtd.n_space)
t = np.linspace(0,T_max,fdtd.nt)
c = fdtd.c
X, Y = np.meshgrid(x, y,indexing="ij")
T,X2,Y2 = np.meshgrid(t,x,y,indexing="ij")
ana_sol = (np.cos(np.sqrt(2)*T*c*np.pi))*np.sin(np.pi*X2)*np.sin(np.pi*Y2)

fdtd.animSurface(-1,1)
fdtd.animContour(-1,1)


err = np.linalg.norm(ana_sol-fdtd.Ez)/np.linalg.norm(ana_sol)
print(f"Relative error in L2 norm: {err*100:.1f} %")

# A bit ugly
fdtd.Ez = ana_sol

fdtd.animSurface(-1,1)
fdtd.animContour(-1,1)


