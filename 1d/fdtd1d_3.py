import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import hilbert, medfilt
from scipy.ndimage import median_filter

"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""

# Units
t_unit = 1E-9  # ns
x_unit = 1E-3  # mm

xmax = 300E-3/x_unit
Tmax = 10E-9/t_unit
dx = 1.5E-3/x_unit
# courant_number = c*dt/dx
courant_number = 1/2

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = eps_0_SI/t_unit**4*x_unit**3
mu_0 = mu_0_SI*t_unit**2/x_unit


# Pulsation for source and epsR
ws = 1E10*2*np.pi*t_unit
wm = 1E9*2*np.pi*t_unit

# Modulation depth
b = 0.67

# Initial relative electric permitivity in the stab
epsR_0 = 3


def epsR_func(t):
    return epsR_0*(1+b*np.sin(wm*t))


# Length of the dielectric stab
v0 = (mu_0*eps_0*epsR_0)**(-1/2)

L = 93E-3/x_unit

# Position of the dielectric
shift = int(L/dx)
k1 = int(xmax/(2*dx))-shift
k2 = k1+shift

# Source and its position
ks = 10


def source_func(t):
    return np.sin(ws*t)


def FDTD_1D(Tmax, courant_number, dx, xmax, epsR_func, k1, k2, source_func, ks):

    c = (eps_0*mu_0)**(-1/2)
    dt = courant_number*dx/c
    t = np.arange(0, Tmax, dt)
    nt = len(t)

    source = source_func(t)
    epsR = epsR_func(t)

    x = np.arange(0, xmax, dx)
    nx = len(x)

    H = np.zeros((nt, nx))
    E = np.zeros((nt, nx))

    ca = np.ones((nx,))
    cb = courant_number*np.ones((nx,))

    E[0, ks] = source[0]
    k_abc = (courant_number-1)/(courant_number+1)

    for i in range(nt-1):

        # Coefficients
        ca[(k1+1):(k2-1)] = epsR[i]/epsR[i+1]
        cb[(k1+1):(k2-1)] = courant_number/epsR[i+1]
        ca[k1] = ca[k2] = (1+epsR[i])/(1+epsR[i+1])
        cb[k1] = cb[k2] = (2*courant_number)/(1+epsR[i+1])

        # Update equations
        H[i+1, 1:] = H[i, 1:]+courant_number*(E[i, 1:]-E[i, :-1])
        E[i+1, :-1] = ca[:-1]*E[i, :-1]+cb[:-1]*(H[i+1, 1:]-H[i+1, :-1])

        # ABC (from Liu phd thesis)

        E[i+1, 0] = E[i, 1]+k_abc*(E[i+1, 1]-E[i, 0])
        E[i+1, -1] = E[i, -2]+k_abc*(E[i+1, -2]-E[i, -1])

        # Soft source
        E[i+1, ks] += source[i+1]

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
    plt.fill_betweenx([-10, 10], x1=x[k1],
                      x2=x[k2], color="grey", alpha=0.8)
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


def plot_E(E, ko):
    plt.plot(t, E[:, ko])
    plt.title(
        "Normalized Electric Field $\\tilde{E}_z(x_0,t)$ with $x_0=x_{out}+2\Delta x$")
    plt.xlabel("time (ns)")
    plt.ylabel("$\\tilde{E}_z(x_0,t)$ (A/m)")
    plt.grid(ls="--")
    plt.show()


def w(Tmax, courant_number, dx, xmax, epsR_func, k1, k2, ks, ws, ko):
    def source1_func(t): return np.sin(ws*t)
    def source2_func(t): return np.cos(ws*t)
    t, _, E1, _ = FDTD_1D(Tmax, courant_number, dx, xmax,
                          epsR_func, k1, k2, source1_func, ks)
    _, _, E2, _ = FDTD_1D(Tmax, courant_number, dx, xmax,
                          epsR_func, k1, k2, source2_func, ks)
    phi = np.arctan2(E1[:, ko], E2[:, ko])
    dt = t[1]-t[0]
    w_vec = (phi[:-4]-8*phi[1:-3]+8*phi[3:-1]-phi[4:])/(12*dt)
    w_vec2 = (phi[1:]-phi[:-1])/dt
    w_vec3 = np.gradient(phi, t)

    return (phi, w_vec, w_vec2, w_vec3, E1, E2)


t, x, E, H = FDTD_1D(Tmax, courant_number, dx, xmax,
                     epsR_func, k1, k2, source_func, ks)


def sec2(t): return 1/np.cos(t)**2


c1 = np.sqrt(4-b**2)
c2 = (2*np.tan(wm*t/2)+b)/c1
c3 = np.arctan(c2)-wm*L*c1
f1 = sec2(wm*t/2)/(1+c2**2)
f2 = ws*(sec2(c3))/(1+(c1/2*np.tan(c3)-b/2)**2)
f_ext_thick = f1*f2/(2*np.pi)

# Compute the extant instantenious frequ

f_ext_thin = 1/(2*np.pi)*ws*(1-(b*L/(2*v0)) *
                             np.cos(wm*t) * (1+b*np.sin(wm*t))**(-1/2))

f_ext = f_ext_thin if L == 3E-3/x_unit else f_ext_thick
# f_ext = f_ext_thin
# f_ext = f_ext_thick
phi, w_vec, w_vec2, w_vec3, E1, E2 = w(Tmax, courant_number, dx, xmax,
                                       epsR_func, k1, k2, ks, ws, k2+2)

# plt.plot(t, phi)
# plt.title("$\\varphi$")
# plt.xlabel("Time (ns)")
# plt.ylabel("Angle(rad)")
# plt.show()

# plt.plot(t[2:-2], w_vec/(2*np.pi))
# plt.title("$f_{FDTD}$ Liu")
# plt.show()

# plt.plot(t[1:], w_vec2/(2*np.pi))
# plt.title("$f_{FDTD}$ df simple")
# plt.show()

# plt.plot(t[1:], w_vec2/(2*np.pi))
# plt.title("$f_{FDTD}$ numpy grad")
# plt.show()

plot_E(E, k2+2)
anim2_E_H(t, x, E, H, k1, k2, -5, 5, 1E-9, 1, 0)

ana_signal = hilbert(E[:, k2+2])
phi= np.unwrap(np.angle(ana_signal))
f_hilbert = (1/(np.pi*2)*np.gradient(phi, t))
# plt.show()
# plt.plot(t,instantaneous_phase)
plt.show()

dt = t[1]-t[0]
i2 = int(2/dt)
i9 = int(9/dt)
f_fdtd = median_filter(w_vec[i2:i9], size=100)/(2*np.pi)
plt.plot(t[i2:i9], f_hilbert[i2:i9],
         label="$f_{Hilbert}$")
plt.plot(t[i2:i9], f_ext[i2:i9], label="$f_{ext}$")
plt.plot(t[i2:i9], f_fdtd, label="$f_{Liu}$")
plt.xlabel("Time (ns)")
plt.ylabel("Frequency (GHz)")
plt.grid(ls="--")
plt.legend()
plt.show()
