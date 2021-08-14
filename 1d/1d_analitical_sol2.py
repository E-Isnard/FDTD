from fdtd import FDTD
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
# jv(n,x) = Bessel Function of 1st kind of order n
L = np.pi
T_max = 1
delta = L/100
d = 1
cfl=1
dt = cfl*delta
def source_func(t): return 0


source_pos = L/2
L_slab = L
slab_pos = 0

def f(t):
    
    return (jv(1,2*np.sqrt(1+t))/np.sqrt(1+t))/jv(1,2)

def g(t):
    return (-jv(0,2*np.sqrt(1+t)))/jv(1,2)


def epsR_func1(t):
    return (1+t)


def J_func1(T, X):
    return np.zeros(T.shape)


def H0_func1(x):
    return np.cos(x)*g(dt/2)


def E0_func1(x):
    return np.sin(x)*f(0)

n5 = -1
def norm(F):
    # return np.max(np.linalg.norm(F, axis=1)*delta)
    return np.linalg.norm(F[n5])

fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func1,
            E0_func1, H0_func1, J_func=J_func1, eps_0=1, mu_0=1, boundary_condition="PEC",cfl=cfl)
fdtd.run()

x = np.linspace(0, L, fdtd.n_space)
x2 = x[:-1]+fdtd.delta/2
t = np.linspace(0, T_max, fdtd.nt)
t2 = t-dt/2
X, T = np.meshgrid(x, t)
X2, T2 = np.meshgrid(x2, t2)

E_ext = f(T)*np.sin(X)
H_ext = g(T2)*np.cos(X2)
plt.plot(x, fdtd.Ez[n5, :], label="E")
plt.plot(x, E_ext[n5, :], label="$E_{ext}$")
plt.legend()
plt.title(
    "Comparison between E and $E_{ext}$ with $\delta = \pi\cdot10^{-3}$ for the $1^{st}$ case")
plt.show()

plt.plot(x2, fdtd.Hy[n5, :], label="H")
plt.plot(x2, H_ext[n5, :], label="$H_{ext}$")
plt.legend()
plt.title(
    "Comparison between H and $H_{ext}$ with $\delta = \pi\cdot10^{-3}$ for the $1^{st}$ case")
plt.show()

err = norm(fdtd.Ez-E_ext)
err2 = norm(fdtd.Hy-H_ext)
rel_err = err/norm(E_ext)
rel_err2 = err2/norm(H_ext)
print("Errors for 1st case (uniform norm):")
print(f"err E = {err:e}")
print(f"err H = {err2:e}")
print(f"rel_err E = {rel_err*100:.2f} %")
print(f"rel_err H = {rel_err2*100:.2f} %\n")