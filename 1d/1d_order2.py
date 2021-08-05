
import numpy as np
from FDTD import FDTD, progress
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.integrate import simps
from scipy.special import jv

L = np.pi
T_max = 10
d = 1


def source_func(t): return 0


source_pos = L/2
L_slab = L
slab_pos = 0


def f(t):
    return (jv(1, 2*np.sqrt(1+t))/np.sqrt(1+t))/jv(1, 2)

def g(t):
    return (-jv(0, 2*np.sqrt(1+t)))/jv(1, 2)

def epsr_func(t):
    return (1+t)

def J_func(T, X):
    return np.zeros(T.shape)

def H0_func(x):
    return np.cos(x)*g(dt/2)

def E0_func(x):
    return np.sin(x)*f(0)

def error(delta, dt):
    cfl = dt/delta
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos,
                epsr_func, E0_func=E0_func, H0_func=H0_func, J_func=J_func, boundary_condition="PEC", eps_0=1, mu_0=1, cfl=cfl, memory_limit=0.7)
    fdtd.run(False, False)
    # fdtd.anim1d(-3/2,3/2)
    x = np.linspace(0, L, fdtd.n_space)
    x2 = x[:-1]+fdtd.delta/2
    t = np.linspace(0, T_max, fdtd.nt)
    t2 = t-fdtd.dt/2
    X, T = np.meshgrid(x, t)
    X2, T2 = np.meshgrid(x2, t2)
    E_ext = np.sin(X)*f(T)
    H_ext = np.cos(X2)*g(T2)
    print(x[-1])
    print(t[-1])
    print(f"{cfl = }")
    print(f"nt = {fdtd.nt}")
    print(f"n_space = {fdtd.n_space}")
    print(f"δ = {fdtd.delta}")
    print(f"{x[1]-x[0] = }")
    print(f"dt = {fdtd.dt}")
    print(f"{t[1]-t[0] = }")
    print(f"||E_ext(0)-E(0)|| = {np.linalg.norm(E_ext[0]-fdtd.Ez[0])}")
    print(f"||H_ext(0)-H(0)|| = {np.linalg.norm(H_ext[0]-fdtd.Hy[0])}")
    print("==========================")

    # energy = fdtd.energy()
    # energy_ext = np.pi/4*(np.exp(t)*np.cos(t)**2+np.sin(t)**2)
    err = np.linalg.norm(E_ext-fdtd.Ez, axis=1)*np.sqrt(delta)
    return (t, err)

delta = L/5
dt = T_max/100000
delta_vec = [delta, delta/2, delta/4, delta/8]
# delta_vec = np.linspace(delta,delta/8,num=100)
err_vec = []
# energy_vec = []
t_vec = []
for i, delta in enumerate(delta_vec):
    t, err = error(delta, dt)
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
err_max = np.max(err_vec, axis=1)
plt.loglog(delta_vec, err_max)
plt.title("Errors as $\delta$ gets smaller")
plt.xlabel("$\delta$")
plt.ylabel("Error")
plt.grid(which="both", ls="--")
plt.show()

m = LinearRegression()
m.fit(np.log(delta_vec.reshape(-1, 1)), np.log(err_max))
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
