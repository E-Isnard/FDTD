from fdtd import FDTD
import numpy as np
import matplotlib.pyplot as plt

L = np.pi
T_max = 10
delta = L/1000
d = 1
cfl=1/2
dt = delta*cfl
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
    return 0


def E0_func2(x):
    return np.sin(x)

n5 = int(5/dt)
def norm(F):
    # return np.max(np.linalg.norm(F, axis=1)*delta)
    return np.linalg.norm(F[n5])


fdtd = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func1,
            E0_func1, H0_func1, J_func=J_func1, eps_0=1, mu_0=1, boundary_condition="PEC",cfl=cfl)
fdtd.run()

x = np.linspace(0, L, fdtd.n_space)
x2 = x[:-1]+delta/2
t = np.linspace(0, T_max, fdtd.nt)
t2 = t
X, T = np.meshgrid(x, t)
X2, T2 = np.meshgrid(x2, t2)
E_ext = 1/np.sqrt((1+T)**3)*np.cos(np.sqrt(3)*np.log(1+T)/2)*np.sin(X)
H_ext = 1/(2*np.sqrt(1+T2))*(-np.cos(np.sqrt(3)*np.log(1+T2)/2) +
                            np.sqrt(3)*np.sin(np.sqrt(3)*np.log(1+T2)/2))*np.cos(X2)

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


fdtd2 = FDTD(L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsR_func2,
             E0_func2, H0_func2, J_func=J_func2, boundary_condition="PEC", eps_0=1, mu_0=1,cfl=cfl)

E_ext2 = np.cos(T)*np.sin(X)
H_ext2 = np.sin(T2)*np.cos(X2)


fdtd2.run()
# print(fdtd.J)

plt.plot(x, fdtd2.Ez[n5, :], label="E")
plt.plot(x, E_ext2[n5, :], label="$E_{ext}$")
plt.title(
    "Comparison between E and $E_{ext}$ with $\delta = \pi\cdot10^{-3}$ for the $2^{nd}$ case")
plt.legend()
plt.show()

plt.plot(x2, fdtd2.Hy[n5, :], label="H")
plt.plot(x2, H_ext2[n5, :], label="$H_{ext}$")
plt.title(
    "Comparison between H and $H_{ext}$ with $\delta = \pi\cdot10^{-3}$ for the $2^{nd}$ case")
plt.legend()
plt.show()
err = norm(fdtd2.Ez-E_ext2)
err2 = norm(fdtd2.Hy-H_ext2)
rel_err = err/norm(E_ext2)
rel_err2 = err2/norm(H_ext2)
print("Errors for 2nd case (uniform norm):")
print(f"err E = {err:e}")
print(f"err H = {err2:e}")
print(f"rel_err E = {rel_err*100:.2f} %")
print(f"rel_err H = {rel_err2*100:.2f} %\n")

