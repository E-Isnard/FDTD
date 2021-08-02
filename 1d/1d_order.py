# from FDTD import FDTD,progress
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression

# L = np.pi
# T_max = 10
# d = 1
# def source_func(t): return 0


# source_pos = L/2
# L_slab = L
# slab_pos = 0


# def epsR_func1(t):
#     return (1+t)**2

# def J_func1(T, X):
#     return np.zeros(T.shape)

# def H0_func1(x):
#     return -np.cos(x)/2

# def E0_func1(x):
#     return np.sin(x)

# def epsR_func2(t):
#     return np.exp(t)

# def J_func2(T, X):
#     return (np.exp(T)*(np.sin(T)-np.cos(T))-np.sin(T))*np.sin(X)

# def H0_func2(x):
#     # return np.zeros(len(x))
#     return np.sin(delta*cfl/2)*np.cos(x+delta/2)

# def E0_func2(x):
#     return np.sin(x)

# def norm(u):
#     return np.linalg.norm(u)*np.sqrt(delta)

# delta = L/10
# cfl_vec = [1/5,1/4,1/3,1/2,1]
# err_vec = []
# dt_vec = []
# print("Compute errors with constant δ")
# for i,cfl in enumerate(cfl_vec):
#     fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
#                 E0_func2, H0_func2, J_func = J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
#     fdtd.run(False, False)
#     x = np.linspace(0, L, fdtd.n_space)
#     t = np.linspace(0, T_max, fdtd.nt)
#     X, T = np.meshgrid(x, t)
#     E_ext = np.cos(T)*np.sin(X)
#     H_ext = np.sin(T)*np.cos(X)
#     err = norm(fdtd.Ez-E_ext)
#     dt_vec.append(fdtd.dt)
#     err_vec.append(err)
#     progress(i,len(cfl_vec))

# plt.plot(dt_vec, err_vec)
# plt.title("Error as dt gets smaller with constant $\delta$")
# plt.grid(ls="--",which="both")
# plt.xlabel("$\mathrm{d}$t [s]")
# plt.ylabel("Error ($L^\infty$)")
# plt.show()

# m = LinearRegression()
# m.fit(np.log(np.array(dt_vec).reshape(-1,1)),np.log(err_vec))
# qt = m.coef_[0]
# print(f"{qt = }")

# delta_0 = 1e-2
# delta_vec = np.array([delta_0,delta_0/2,delta_0/4,delta_0/8])
# dt = 1e-3/4
# err_vec = []
# # cfl=1
# print("Computing errors with constant dt")
# for i,delta in enumerate(delta_vec):
#     cfl = fdtd.c*dt/delta
#     fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
#                 E0_func2, H0_func2, J_func = J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
#     fdtd.run(False, False)
#     x = np.linspace(0, L, fdtd.n_space)
#     t = np.linspace(0, T_max, fdtd.nt)
#     X, T = np.meshgrid(x, t)
#     E_ext = np.cos(T)*np.sin(X)
#     H_ext = np.sin(T)*np.cos(X)
#     err  = norm(fdtd.Ez-E_ext)
#     err_vec.append(err)
#     progress(i,len(delta_vec))

# plt.loglog(delta_vec, err_vec)
# plt.grid(ls="--",which="both")
# plt.title("Error as $\delta$ gets smaller with constant dt")
# plt.xlabel("$\delta$ [m]")
# plt.ylabel("Error ($L^\infty$)")
# plt.show()
# m.fit(np.log(delta_vec.reshape(-1,1)),np.log(err_vec))
# qs = m.coef_[0]
# print(f"{qs = }")

import numpy as np
from FDTD import FDTD, progress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.integrate import simps

L = np.pi
T_max = 20
d = 1

def source_func(t): return 0

source_pos = L/2
L_slab = L
slab_pos = 0

def E0_func(x):
    return np.sin(x)

def H0_func(x):
    return -np.cos(x)/2
    
def J_func(T,X):
    # return (np.exp(T)*(np.sin(T)-np.cos(T))-np.sin(T))*np.sin(X)
    return 0*T

def epsr_func(t):
    return (1+t)**2

def error(delta,dt):
    cfl = dt/delta
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos,
                epsr_func, E0_func=E0_func,H0_func=H0_func,J_func=J_func, boundary_condition="PEC", eps_0=1, mu_0=1, cfl=cfl,memory_limit=0.7)
    fdtd.run(False,False)
    # fdtd.anim1d(-3/2,3/2)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)
    print(x[-1])
    print(t[-1])
    print(f"{cfl = }")
    print(f"nt = {fdtd.nt}")
    print(f"n_space = {fdtd.n_space}")
    print(f"δ = {fdtd.delta}")
    print(f"{x[1]-x[0] = }")
    print(f"dt = {fdtd.dt}")
    print(f"{t[1]-t[0] = }")
    print("==========================")
    
    X, T = np.meshgrid(x, t)
    E_ext = np.sin(X)*np.cos(T)
    H_ext = np.cos(X)*np.sin(T)
    # energy = fdtd.energy()
    # energy_ext = np.pi/4*(np.exp(t)*np.cos(t)**2+np.sin(t)**2)
    err = np.linalg.norm(E_ext-fdtd.Ez, axis=1)*np.sqrt(delta)
    return (t,err)

delta = np.pi/10
dt = 1e-3
delta_vec = [delta,delta/2,delta/4,delta/8]
err_vec = []
# energy_vec = []
t_vec = []
for i,delta in enumerate(delta_vec):
    t,err = error(delta,dt)
    err_vec.append(err)
    # energy_vec.append(energy)
    t_vec.append(t)

err_vec = np.array(err_vec)
delta_vec = np.array(delta_vec)
# energy_vec = np.array(energy_vec)
plt.plot(t_vec[0], err_vec[0], label="L2 error w/ $\delta$")
plt.plot(t_vec[1], err_vec[1], label="L2 error2 w/ $\delta/2$")
plt.plot(t_vec[2], err_vec[2], label="L2 error3 w/ $\delta/4$")
plt.plot(t_vec[3], err_vec[3], label="L2 error3 w/ $\delta/8$")
plt.xlabel("t")
plt.ylabel("$||E-E_{ext}||_2$")
plt.legend()
plt.show()
err_max = np.max(err_vec,axis=1)
plt.loglog(delta_vec,err_max)
plt.title("Errors as $\delta$ gets smaller")
plt.xlabel("$\delta$")
plt.ylabel("Error")
plt.grid(which="both",ls="--")
plt.show()

m = LinearRegression()
m.fit(np.log(delta_vec.reshape(-1,1)),np.log(err_max))
r = m.coef_[0]
print(f"{r = }")

# plt.plot(t_vec[0], energy_vec[0], label="$\mathcal{E}_{\delta}$")
# plt.plot(t_vec[1], energy_vec[1], label="$\mathcal{E}_{\delta/2}$")
# plt.plot(t_vec[2], energy_vec[2], label="$\mathcal{E}_{\delta/4}$")
# plt.plot(t_vec[3], energy_vec[3], label="$\mathcal{E}_{\delta/8}$")
# plt.plot(t_vec[-1],energy_ext , label="$\mathcal{E}_{ext}$")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel("$\mathcal{E}(t)$")
# plt.title("Numerical Energy")
# plt.show()

# err_energy = np.max(np.abs(energy_ext-energy_vec),axis=1)
# plt.loglog(delta_vec,err_energy)
# plt.title("Differences in energy as $\delta$ gets smaller")
# plt.xlabel("$\delta$")
# plt.ylabel("$\max|\mathcal{E}-\mathcal{E}_{ext}|$")
# plt.grid(ls="--",which="both")

# plt.show()
# m.fit(np.log(delta_vec.reshape(-1,1)),np.log(err_energy))
# r_energy = m.coef_[0]
# print(f"{r_energy = }")

