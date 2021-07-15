import numpy as np
from FDTD import FDTD
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec

L=1
T_max = 10
d = 1

def source_func(t):return 0
source_pos = L/2
L_slab  =L
slab_pos = 0


def E0_func(x):
	return np.sin(np.pi*x)

def epsr_func(t):
	return 1


def norm(u):
    return np.norm(u)

delta = 5e-3
fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func,E0_func=E0_func,boundary_condition="PEC",eps_0=1,mu_0=1)
fdtd.run()
x = np.linspace(0, L, fdtd.n_space)
t = np.linspace(0, T_max, fdtd.nt)
X, T = np.meshgrid(x, t)
E_ext = np.sin(np.pi*X)*np.cos(np.pi/L*T)
H_ext = np.cos(np.pi*X)*np.sin(np.pi/L*T)
# # fdtd.Ez = E_ext
# # fdtd.Hy = H_ext
# # fdtd.anim1d(-3/2,3/2)
# err = np.linalg.norm(E_ext-fdtd.Ez,axis=1)

# delta/=2
# fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func,E0_func=E0_func,boundary_condition="PEC",eps_0=1,mu_0=1)
# fdtd.run()
# x = np.linspace(0, L, fdtd.n_space)
# t2 = np.linspace(0, T_max, fdtd.nt)
# X, T = np.meshgrid(x, t2)
# E_ext = np.sin(np.pi*X)*np.cos(np.pi/L*T)
# # fdtd.anim1d(-3/2,3/2)
# err2 = np.linalg.norm(E_ext-fdtd.Ez,axis=1)


# delta/=2
# fdtd = FDTD(L,delta,T_max,d,source_func,source_pos,L_slab,slab_pos,epsr_func,E0_func=E0_func,boundary_condition="PEC",eps_0=1,mu_0=1)
# fdtd.run()
# x = np.linspace(0, L, fdtd.n_space)
# t3 = np.linspace(0, T_max, fdtd.nt)
# X, T = np.meshgrid(x, t3)
# E_ext = np.sin(np.pi*X)*np.cos(np.pi/L*T)
# # fdtd.anim1d(-3/2,3/2)
# err3 = np.linalg.norm(E_ext-fdtd.Ez,axis=1)

# plt.plot(t,err,label="L2 error with $\delta=5\cdot10^{-3}$")
# plt.plot(t2,err2,label="L2 error with $\delta=2.5\cdot10^{-3}$")
# plt.plot(t3,err3,label="L2 error with $\delta=1.25\cdot10^{-3}$")
# plt.xlabel("t")
# plt.ylabel("$||E-E_{ext}||_2$")
# plt.legend()
# plt.show()


U = 1/2*(fdtd.Ez**2+fdtd.Hy**2)
U_ext = 1/2*(E_ext**2+H_ext**2)
U_func = lambda x : U[:,int(x/fdtd.delta)]
U_ext_func = lambda x : U_ext[:,int(x/fdtd.delta)]
energy,_ = quad_vec(U_func,0,1)
energy_ext,_ = quad_vec(U_ext_func,0,1)


plt.plot(t,energy,label="Numerical energy")
plt.plot(t,energy_ext,label="Exact energy")
plt.legend()
plt.ylabel("$\mathcal{E}(t)$")
plt.xlabel("t")
plt.show()


