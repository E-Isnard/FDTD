"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation



# Units
t_unit = 1E-9  # ns
x_unit = 1E-3  # mm

xmax = np.pi
Tmax = 10
dx = 0.01
# courant_number = c*dt/dx
courant_number = 0.1

# EM constants
eps_0 = 1
mu_0 = 1


def epsR_func(t):
    return (1+t)**2

def FDTD_1D(Tmax, courant_number, dx, xmax, epsR_func):

    c = (eps_0*mu_0)**(-1/2)
    dt = courant_number*dx/c
    t = np.arange(0, Tmax, dt)
    nt = len(t)

    epsR = epsR_func(t)
    x = np.arange(0, xmax, dx)
    nx = len(x)

    H = np.zeros((nt, nx))
    E = np.zeros((nt, nx))

    E[0, :] = np.sin(x)
    H[0,:] = -np.cos(x)/2

    for i in range(nt-1):

        # Coefficients
        ca = epsR[i]/epsR[i+1]
        cb = courant_number/epsR[i+1]

        # Update equations
        H[i+1, 1:] = H[i, 1:]+courant_number*(E[i, 1:]-E[i, :-1])
        E[i+1, :-1] = ca*E[i, :-1]+cb*(H[i+1, 1:]-H[i+1, :-1])

        # ABC (from Liu phd thesis)

        E[i+1, 0] = 0
        E[i+1, -1] = 0
        H[i+1,0] = H[i+1,1]

    return (t, x, E, H)


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
                     epsR_func)


def sec2(t): return 1/np.cos(t)**2
k1=0
k2=0

X,T = np.meshgrid(x,t)
E_ext = 1/np.sqrt((1+T)**3)*np.cos(np.sqrt(3)*np.log(1+T)/2)*np.sin(X)
H_ext = 1/(2*np.sqrt(1+T))*(-np.cos(np.sqrt(3)*np.log(1+T)/2)+np.sqrt(3)*np.sin(np.sqrt(3)*np.log(1+T)/2))*np.cos(X)
anim2_E_H(t, x, E, H, k1, k2, -5, 5, 1E-9, 1, 0)
anim2_E_H(t, x, E_ext, H_ext, k1, k2, -5, 5, 1E-9, 1, 0)


dt = t[1]-t[0]
n5 = int(5/dt)
err = np.linalg.norm(E[n5,:]-E_ext[n5,:])
err2 = np.linalg.norm(H[n5,:]-H_ext[n5,:])
plt.plot(x,E[n5,:],label="E")
plt.plot(x,E_ext[n5,:],label="E_ext")
plt.legend()
plt.show()

plt.plot(x,H[n5,:])
plt.plot(x,H_ext[n5,:])
plt.show()
print(f"err  = {err:e}")
print(f"err2 = {err2:e}")
