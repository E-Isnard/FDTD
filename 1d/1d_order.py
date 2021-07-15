from FDTD import FDTD,progress
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

def norm(u):
    return np.max(np.max(np.abs(u),axis=1))

delta = 1e-1
cfl_vec = np.logspace(0,-1,num=200,base=2)
err_vec = []
dt_vec = []
print("Compute errors with constant δ")
for i,cfl in enumerate(cfl_vec):
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
                E0_func2, H0_func2, J_func = J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
    fdtd.run(False, False)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)
    X, T = np.meshgrid(x, t)
    E_ext = np.cos(T)*np.sin(X)
    H_ext = np.sin(T)*np.cos(X)
    err = norm(fdtd.Ez-E_ext)
    dt_vec.append(fdtd.dt)
    err_vec.append(err)
    progress(i,len(cfl_vec))

plt.plot(dt_vec, err_vec)
plt.title("Error as dt gets smaller with constant $\delta$")
plt.grid(ls="--",which="both")
plt.xlabel("$\mathrm{d}$t [s]")
plt.ylabel("Error ($L^\infty$)")
plt.show()

qt = np.log(err_vec[-1]/err_vec[0])/np.log(dt_vec[-1]/dt_vec[0])
print(f"{qt = }")

delta_vec = np.logspace(-1,-3,num=200,base=2)
dt = 1e-2
err_vec = []
# cfl=1
print("Computing errors with constant dt")
for i,delta in enumerate(delta_vec):
    cfl = fdtd.c*dt/delta
    fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
                E0_func2, H0_func2, J_func = J_func2, cfl=cfl, boundary_condition="PEC", eps_0=1, mu_0=1)
    fdtd.run(False, False)
    x = np.linspace(0, L, fdtd.n_space)
    t = np.linspace(0, T_max, fdtd.nt)

    X, T = np.meshgrid(x, t)
    E_ext = np.cos(T)*np.sin(X)
    H_ext = np.sin(T)*np.cos(X)
    err  = norm(fdtd.Ez-E_ext)
    err_vec.append(err)
    if i%20==0:
        print(norm(fdtd.Ez-E_ext))
    progress(i,len(delta_vec))

plt.plot(delta_vec, err_vec)
plt.grid(ls="--",which="both")
plt.title("Error as $\delta$ gets smaller with constant dt")
plt.xlabel("$\delta$ [m]")
plt.ylabel("Error ($L^\infty$)")
plt.show()
qs = np.log(err_vec[-1]/err_vec[0])/np.log(delta_vec[-1]/delta_vec[0])
print(f"{qs = }")
