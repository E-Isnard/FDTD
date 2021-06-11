import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
1D FDTD with time and space dependant electric permitivity

H has only one component along y and E only along z

H_y and E_z only depend on x.

We use a normalized version of E: Ẽ_z = sqrt(eps/mu)*E

"""

t_unit = 1E-9
x_unit = 1E-3

xmax = 300E-3/x_unit
Tmax = 6E-9/t_unit
dx = 1.5E-3/x_unit
# courant_number = c*dt/dx
courant_number = 1/2

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = eps_0_SI/t_unit**4*x_unit**3
mu_0 = mu_0_SI*t_unit**2/x_unit
epsR_0 = 3

# Pulsation for source and epsR
ws = 1E10*2*np.pi*t_unit
wm = 1E9*2*np.pi*t_unit

# Length of the dielectric stab
L = 93E-3/x_unit

# Source and its position
ks = 10
def source_func(t):
    return np.sin(ws*t)
# Modulation depth
b = 0.67


def FDTD_1D(Tmax, dx, xmax, epsR_0, L, source_func, ks, b):

    c = (eps_0*mu_0)**(-1/2)
    dt = courant_number*dx/c
    t = np.arange(0, Tmax, dt)
    nt = len(t)

    source = source_func(t)

    x = np.arange(0, xmax, dx)
    nx = len(x)
    epsR = epsR_0*(1+b*np.sin(wm*t))

    # Position of the dielectric
    shift = int(L/dx)
    k1 = int(nx/2)-shift
    k2 = k1+shift

    H = np.zeros((nt, nx))
    E = np.zeros((nt, nx))

    ca = np.ones((nx,))
    cb = courant_number*np.ones((nx,))

    E[0, ks] = source[0]

    for i in range(nt-1):

        # Coefficients
        ca[(k1+1):(k2-1)] = epsR[i]/epsR[i+1]
        cb[(k1+1):(k2-1)] = courant_number/epsR[i+1]
        ca[k1] = ca[k2] = (1+epsR[i])/(1+epsR[i+1])
        cb[k1] = cb[k2] = (2*courant_number)/(1+epsR[i+1])

        # Update equations
        H[i+1, :-1] = H[i, :-1]+courant_number*(E[i, 1:]-E[i, :-1])
        E[i+1, 1:] = ca[1:]*E[i, 1:]+cb[1:]*(H[i+1, 1:]-H[i+1, :-1])

        # ABC (from Liu phd thesis)
        k_abc = (courant_number-1)/(courant_number+1)
        E[i+1, 0] = E[i, 1]+k_abc*(E[i+1, 1]-E[i, 0])
        E[i+1, -1] = E[i, -2]+k_abc*(E[i+1, -2]-E[i, -1])

        # Soft source
        E[i+1, ks] += source[i+1]

    return (t, x, k1, k2, E, H)


def anim_E_H(t, E, H, k1, k2, y_low, y_high, anim=True, interval=1E-3):
    for i in range(len(t)):
        y = H[i]
        y2 = E[i]
        if i == 0:
            line, = plt.plot(x, y)
            line2, = plt.plot(x, y2)
            plt.title("Propagation of $\\tilde{E}_z$ and $H_y$")
            plt.xlabel("x (mm)")
            plt.ylabel("Amplitude (A/m)")
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


def plot_E(E, ko):

    plt.plot(t, E[:, ko])
    plt.title(
        "Normalized Electric Field $\\tilde{E}_z(x_0,t)$ with $x_0=x_{out}+2\Delta x$")
    plt.xlabel("time (ns)")
    plt.ylabel("$\\tilde{E}_z(x_0,t)$ (A/m)")
    plt.show()


def w(Tmax, dx, xmax, epsR_0, L, ks, b, ws, ko):
    def source1_func(t): return np.sin(ws*t)
    def source2_func(t): return np.cos(ws*t)
    t, _, _, _, E1, _ = FDTD_1D(
        Tmax, dx, xmax, epsR_0, L, source1_func, ks, b)
    _, _, _, _, E2, _ = FDTD_1D(
        Tmax, dx, xmax, epsR_0, L, source2_func, ks, b)
    nt = len(t)
    phi = np.arctan2(E1[:, ko], E2[:, ko])
    dt = Tmax/nt
    w_vec = (phi[:-4]-8*phi[1:-3]+8*phi[3:-1]-phi[4:])/(12*dt)

    return (phi, w_vec, E1, E2)


t, x, k1, k2, E, H = FDTD_1D(Tmax, dx, xmax, epsR_0, L, source_func, ks, b)

# coeff = (2*np.tan(wm*t/2)+b)/np.sqrt(4-b**2)
# coeff2 = wm*L*np.sqrt(4-b**2)
# coeff3 = np.arctan(coeff)-coeff2
# w_ext = (1/np.cos(wm*t/2)**2)/(1+coeff**2)*ws/np.cos(coeff3**2)**2 * 1/(1+(np.sqrt(4-b**2)/2*np.tan(coeff3)-b/2)**2)

# v0 = 1/(np.sqrt(epsR_0*mu_0*eps_0))
# w_ext = ws*(1-(b*L/(2*v0))*np.cos(wm*t)/np.sqrt(1+b*np.sin(wm*t)))
# phi, w_vec, E1, E2 = w(Tmax, dx, xmax, epsR_0, L, ks, b, ws, k2+2)
# plt.plot(t[2:-2],w_vec)
# plt.show()
# plt.plot(t, w_ext/(np.pi*2))
# plt.show()
plot_E(E,k2+2)
anim_E_H(t,E,H,k1,k2,-5,5,1)
