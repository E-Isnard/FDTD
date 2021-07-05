"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

xmax = np.pi
Tmax = 10
dx = 0.005
# courant_number = c*dt/dx
courant_number = 1/2

# EM constants
eps_0 = 1
mu_0 = 1


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


def FDTD_1D(Tmax, courant_number, dx, xmax, epsR_func, E0_func, H0_func, J_func):

    c = (eps_0*mu_0)**(-1/2)
    dt = courant_number*dx/c
    t = np.arange(0, Tmax, dt)
    nt = len(t)

    epsR = epsR_func(t)

    x = np.arange(0, xmax, dx)
    nx = len(x)
    X, T = np.meshgrid(x, t)
    J = J_func(T, X)
    # print(J)
    del X
    del T

    H = np.zeros((nt, nx))
    E = np.zeros((nt, nx))

    E[0, :] = E0_func(x)
    H[0, :] = H0_func(x)

    for i in range(nt-1):

        # Coefficients
        ca = epsR[i]/epsR[i+1]
        cb = courant_number/epsR[i+1]

        # Update equations
        H[i+1, 1:] = H[i, 1:]+courant_number*(E[i, 1:]-E[i, :-1])
        E[i+1, :-1] = ca*E[i, :-1]+cb*(H[i+1, 1:]-H[i+1, :-1]-J[i+1,:-1]*dx)

        # Dirichlet BC

        E[i+1, 0] = 0
        E[i+1, -1] = 0
        H[i+1, 0] = H[i+1, 1]

    return t, x, E, H


def anim_E_H(t, x, E, H, k1, k2, y_low, y_high, anim=True, interval=1E-3):
    for i in range(len(t)):
        y = H[i]
        y2 = E[i]
        if i == 0:
            line, = plt.plot(x, y)
            line2, = plt.plot(x, y2)
            plt.title("Propagation of $\\tilde{E}_z$ and $H_y$")
            plt.xlabel("x (mm)")
            plt.ylabel("Amplitude (A/m)")
            plt.grid(ls="-")
            plt.ylim([y_low, y_high])
            plt.legend(["$H_y$", "$\\tilde{E}_z$"])
            plt.fill_betweenx([y_low, y_high], x1=x[k1],
                              x2=x[k2], color="grey", alpha=0.8)
        else:
            line.set_ydata(y)
            line2.set_ydata(y2)
        if anim:
            plt.pause(interval)
    plt.show()


def anim2_E_H(t, x, E, H, k1, k2, y_low, y_high, interval=1E-3, show=True, save=False):
    nt = len(t)
    nx = len(x)
    fig = plt.figure()
    line, line2 = plt.plot(x, np.array([E[0], H[0]]).T)

    plt.title("Propagation of $\\tilde{E}_z$ and $H_y$")
    plt.xlabel("x (mm)")
    plt.ylabel("Amplitude (A/m)")
    plt.legend(["$H_y$", "$\\tilde{E}_z$"])
    # plt.fill_betweenx([-10, 10], x1=x[k1],
    #   x2=x[k2], color="grey", alpha=0.8)
    plt.ylim(y_low, y_high)
    plt.xlim(0, xmax)

    def animate(i):
        y1 = E[i].reshape((nx, 1))
        y2 = H[i].reshape((nx, 1))
        line.set_data(x, y1)
        line2.set_data(x, y2)

    frames = nt/5 if save else nt
    ani = animation.FuncAnimation(
        fig, animate, interval=interval, frames=frames)

    if save:
        ani.save("anim.gif", fps=60)
    if show:
        plt.show()
    plt.close()


t, x, E, H = FDTD_1D(Tmax, courant_number, dx, xmax,
                     epsR_func1, E0_func1, H0_func1, J_func1)

_,_,E2,H2 = FDTD_1D(Tmax,courant_number,dx,xmax,epsR_func2,E0_func2,H0_func2,J_func2)


X, T = np.meshgrid(x, t)
E_ext = 1/np.sqrt((1+T)**3)*np.cos(np.sqrt(3)*np.log(1+T)/2)*np.sin(X)
H_ext = 1/(2*np.sqrt(1+T))*(-np.cos(np.sqrt(3)*np.log(1+T)/2) +
                            np.sqrt(3)*np.sin(np.sqrt(3)*np.log(1+T)/2))*np.cos(X)


dt = t[1]-t[0]
n5 = int(5/dt)
err = np.linalg.norm(E[n5, :]-E_ext[n5, :])
err2 = np.linalg.norm(H[n5, :]-H_ext[n5, :])
rel_err = err/np.linalg.norm(E_ext[n5,:])
rel_err2 = err2/np.linalg.norm(H_ext[n5,:])
plt.plot(x, E[n5, :], label="E")
plt.plot(x, E_ext[n5, :], label="E_ext")
plt.legend()
plt.show()

plt.plot(x, H[n5, :])
plt.plot(x, H_ext[n5, :])
plt.show()
print("Errors for 1st case (L2 norm):")
print(f"err E = {err:e}")
print(f"err H = {err2:e}")
print(f"rel_err E = {rel_err*100:.2f} %")
print(f"rel_err H = {rel_err2*100:.2f} %\n")


E_ext2 = np.cos(T)*np.sin(X)
H_ext2 = np.sin(T)*np.cos(X)


err = np.linalg.norm(E2[n5, :]-E_ext2[n5, :])
err2 = np.linalg.norm(H2[n5, :]-H_ext2[n5, :])
rel_err = err/np.linalg.norm(E_ext2[n5,:])
rel_err2 = err2/np.linalg.norm(H_ext2[n5,:])
plt.plot(x, E2[n5, :], label="E")
plt.plot(x, E_ext2[n5, :], label="E_ext")
plt.legend()
plt.show()

plt.plot(x, H2[n5, :])
plt.plot(x, H_ext2[n5, :])
plt.show()
print("Errors for 2nd case (L2 norm):")
print(f"err E = {err:e}")
print(f"err H = {err2:e}")
print(f"rel_err E = {rel_err*100:.2f} %")
print(f"rel_err H = {rel_err2*100:.2f} %")