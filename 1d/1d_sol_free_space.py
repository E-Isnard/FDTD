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

def epsr_func(t):
    return 1

def error(delta,dt):
    cfl = dt/delta
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos,
                epsr_func, E0_func=E0_func, boundary_condition="PEC", eps_0=1, mu_0=1, cfl=cfl)
    fdtd.run(False,False)
    print(f"{cfl = }")
    print(f"nt = {fdtd.nt}")
    print(f"n_space = {fdtd.n_space}")
    print("==========================")
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)
    X, T = np.meshgrid(x, t)
    E_ext = np.sin(np.pi*X)*np.cos(np.pi/L*T)
    H_ext = np.cos(np.pi*X)*np.sin(np.pi/L*T)
    energy = fdtd.energy()
    err = np.linalg.norm(E_ext-fdtd.Ez, axis=1)*np.sqrt(delta)
    return (t,err,energy)

delta = 5e-2
dt = 1e-3
delta_vec = [delta,delta/2,delta/4]
err_vec = []
energy_vec = []
t_vec = []
for i,delta in enumerate(delta_vec):
    t,err,energy = error(delta,dt)
    err_vec.append(err)
    energy_vec.append(energy)
    t_vec.append(t)

err_vec = np.array(err_vec)
delta_vec = np.array(delta_vec)
energy_vec = np.array(energy_vec)
plt.plot(t_vec[0], err_vec[0], label="L2 error")
plt.plot(t_vec[1], err_vec[1], label="L2 error2")
plt.plot(t_vec[2], err_vec[2], label="L2 error3")
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

plt.plot(t_vec[0], energy_vec[0], label="$\mathcal{E}_{\delta}$")
plt.plot(t_vec[1], energy_vec[1], label="$\mathcal{E}_{\delta/2}$")
plt.plot(t_vec[2], energy_vec[2], label="$\mathcal{E}_{\delta/4}$")
plt.plot(t_vec[0], np.ones(t_vec[0].shape)*1/4, label="$\mathcal{E}_{ext}$")
plt.legend()
plt.xlabel("t")
plt.ylabel("$\mathcal{E}(t)$")
plt.show()