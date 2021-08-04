import numpy as np
from FDTD import FDTD, progress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

L = 1
T_max = 10
d = 1

def source_func(t): return 0

source_pos = L/2
L_slab = L
slab_pos = 0

def E0_func(x):
    return np.sin(np.pi*x)

def H0_func(x):
    return np.cos(np.pi*x)*np.sin(-np.pi*dt/2)

def epsr_func(t):
    return 1

def error(delta,dt,info=True):
    cfl = dt/delta
    # cfl=1/2
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos,
                epsr_func, E0_func=E0_func,H0_func=H0_func, boundary_condition="PEC", eps_0=1, mu_0=1, cfl=cfl)
    fdtd.run(False,False)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)
    t = np.linspace(0, T_max, fdtd.nt)
    if info:
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
    E_ext = np.sin(np.pi*X)*np.cos(np.pi*T)
    # H_ext = np.cos(np.pi*X)*np.sin(np.pi/L*T)
    # energy = fdtd.energy()
    err = np.linalg.norm(E_ext-fdtd.Ez, axis=1)*np.sqrt(delta)
    return (t,err)

delta = L/5
dt = T_max/100000
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
# plt.plot(t_vec[0], np.ones(t_vec[0].shape)*1/4, label="$\mathcal{E}_{ext}$")
# plt.legend()
# plt.xlabel("t")
# plt.ylabel("$\mathcal{E}(t)$")
# plt.title("Numerical Energy")
# plt.show()

# err_energy = np.max(1/4-energy_vec,axis=1)
# plt.loglog(delta_vec,err_energy)
# plt.title("Differences in energy as $\delta$ gets smaller")
# plt.xlabel("$\delta$")
# plt.ylabel("$\max|\mathcal{E}-\mathcal{E}_{ext}|$")
# plt.grid(ls="--",which="both")

# plt.show()
# m.fit(np.log(delta_vec.reshape(-1,1)),np.log(err_energy))
# r_energy = m.coef_[0]
# print(f"{r_energy = }")