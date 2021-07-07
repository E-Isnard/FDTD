
from FDTD import FDTD, progress
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
T_max = 10
d = 1
def source_func(t): return 0


source_pos = L/2
L_slab = L
slab_pos = 0


def epsR_func1(t):
    return (1+t)**2

def J_func1(T, X):
    return np.zeros(T.shape)

def H0_func1(x):
    return -np.cos(x)/2

def E0_func1(x):
    return np.sin(x)

def epsR_func2(t):
    return np.exp(t)

def J_func2(T, X):
    return (np.exp(T)*(np.sin(T)-np.cos(T))-np.sin(T))*np.sin(X)

def H0_func2(x):
    return np.zeros(len(x))

def E0_func2(x):
    return np.sin(x)

delta = 5e-3
cfl_vec = np.logspace(-1,-3,num=4,base=2)
err_vec = []
dt_vec = []
print("Compute errors with constant δ")
for i,cfl in enumerate(cfl_vec):
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
                E0_func2, H0_func2, J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
    fdtd.run(False, False)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)
    X, T = np.meshgrid(x, t)
    E_ext = np.cos(T)*np.sin(X)
    H_ext = np.sin(T)*np.cos(X)
    n5 = int(5/fdtd.dt)
    err = np.linalg.norm((fdtd.Ez[n5, :]-E_ext[n5, :])/fdtd.n_space)
    dt_vec.append(fdtd.dt)
    err_vec.append(err)
    progress(i,len(cfl_vec))

plt.loglog(dt_vec, err_vec)
plt.title("Error as dt gets smaller with constant $\delta$")
plt.grid(ls="--",which="both")
plt.show()

qt = np.log(err_vec[-1]/err_vec[0])/np.log(dt_vec[-1]/dt_vec[0])
print(f"{qt = }")

delta_vec = np.logspace(-2,-5,num=4,base=2)
dt = 1e-4
err_vec = []
print("Computing errors with constant dt")
for i,delta in enumerate(delta_vec):
    cfl = fdtd.c*dt/delta
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
                E0_func2, H0_func2, J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
    fdtd.run(False, False)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)

    X, T = np.meshgrid(x, t)
    E_ext = np.cos(T)*np.sin(X)
    H_ext = np.sin(T)*np.cos(X)
    n5 = int(5/fdtd.dt)
    mid = int(fdtd.n_space/2)
    err = np.linalg.norm((fdtd.Ez[n5, :]-E_ext[n5, :])/fdtd.n_space)
    err_vec.append(err)
    progress(i,len(delta_vec))

plt.loglog(delta_vec, err_vec)
plt.grid(ls="--",which="both")
plt.title("Error as $\delta$ gets smaller with constant cfl")
plt.show()
qs = np.log(err_vec[-1]/err_vec[0])/np.log(delta_vec[-1]/delta_vec[0])
print(f"{qs = }")
