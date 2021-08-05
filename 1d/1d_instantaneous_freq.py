from FDTD import FDTD, progress
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter


d = 1
L = 800e-3
T_max = 10e-9
delta = 8e-4

source_pos = 10*delta
ws = 1e10*2*np.pi
def source_func1(t): return np.sin(ws*t)
def source_func2(t): return np.cos(ws*t)


wm = 1e9*2*np.pi
b = 0.67
epsr_0 = 3.
def epsr_func(t): return epsr_0*(1+b*np.sin(wm*t))
eps0 = 8.85418782e-12
mu0 = 4e-7*np.pi
v0 = 1/np.sqrt(epsr_0*eps0*mu0)
L0 = (4*np.pi*v0)/(wm*np.sqrt(4-b**2))
L_slab = 3e-3
slab_pos = L/2-L_slab

def f_Liu_func(L_slab):
    slab_pos = L/2-L_slab
    fdtd = FDTD(L, delta, T_max, d, source_func1,
                source_pos, L_slab, slab_pos, epsr_func)

    fdtd.run(progress_bar=False, info=False)
    # fdtd.anim1d(-1,1)
    # fdtd.plotE1d(slab_pos+L_slab+2*delta)

    E1 = fdtd.Ez

    fdtd2 = FDTD(L, delta, T_max, d, source_func2,
                source_pos, L_slab, slab_pos, epsr_func)

    fdtd2.run(progress_bar=False, info=False)

    E2 = fdtd2.Ez
    ko = int((slab_pos+L_slab+2*delta)/delta)
    phi = np.arctan2(E1[:, ko], E2[:, ko])
    f_Liu = (phi[:-4]-8*phi[1:-3]+8*phi[3:-1]-phi[4:])/(12*fdtd.dt*2*np.pi)
    f_Liu = median_filter(f_Liu, size=10)
    return f_Liu

f_Liu = f_Liu_func(L_slab)

fdtd = FDTD(L, delta, T_max, d, source_func1,
                source_pos, L_slab, slab_pos, epsr_func)
fdtd.run(False, False)
f_hilbert = fdtd.instant_freq((slab_pos+L_slab+2*delta))
n2 = int(2e-9/fdtd.dt)
n9 = int(9e-9/fdtd.dt)
t = np.linspace(0, T_max, fdtd.nt)


def sec2(t): return (np.cos(t))**(-2)


def f_ext_func(L_slab):

    v0 = (fdtd.eps_0*fdtd.mu_0*epsr_0)**(-1/2)

    c1 = np.sqrt(4-b**2)
    c2 = (2*np.tan(wm*t/2)+b)/c1
    c3 = np.arctan(c2)-wm*L_slab*c1/(4*v0)
    f1 = sec2(wm*t/2)/(1+c2**2)
    f2 = ws*(sec2(c3))/(1+(c1/2*np.tan(c3)-b/2)**2)
    f_ext = f1*f2/(2*np.pi)

    return f_ext


f_ext = f_ext_func(L_slab)
plt.plot(t[n2:n9]*1e9, f_Liu[n2:n9]/1e9, label="$f_{Liu}$")
plt.plot(t[n2:n9]*1e9, f_hilbert[n2:n9]/1e9, label="$f_{Hilbert}$")
plt.plot(t[n2:n9]*1e9, f_ext[n2:n9]/1e9, label="$f_{ext}$")
plt.xlabel("Time [ns]")
plt.ylabel("Frequency [GHz]")
plt.title(f"Instantaneous frequencies for b={b}")
plt.legend()
plt.show()


swings_ext = []
swings_hilbert = []
swings_Liu = []
L_vec = np.linspace(0, 350e-3, 100)
t = np.linspace(0, T_max, fdtd.nt)

n2 = int(2e-9/fdtd.dt)
n9 = int(9e-9/fdtd.dt)
xo = L_slab+slab_pos+2*fdtd.delta
for i, L_slab in enumerate(L_vec):

    slab_pos = L/2-L_slab
    fdtd.L_slab = L_slab
    fdtd.slab_pos = slab_pos
    fdtd.run(False, False)
    f_hilbert = fdtd.instant_freq(xo)
    f_ext = f_ext_func(L_slab)
    f_Liu = f_Liu_func(L_slab)
    swings_ext.append(np.max(f_ext[n2:n9])-np.min(f_ext[n2:n9]))
    swings_hilbert.append(np.max(f_hilbert[n2:n9])-np.min(f_hilbert[n2:n9]))
    swings_Liu.append(np.max(f_Liu[n2:n9])-np.min(f_Liu[n2:n9]))
    progress(i, 100)

swings_hilbert = np.array(swings_hilbert)
swings_ext = np.array(swings_ext)
swings_Liu = np.array(swings_Liu)
plt.plot(L_vec*1e3, swings_ext/1e9, label="Exact freq swings")
plt.plot(L_vec*1e3, swings_hilbert/1e9, label="Hilbert freq swings")
plt.plot(L_vec*1e3, swings_Liu/1e9, label="Liu freq swings")
plt.legend()
plt.xlabel("Width of slab [mm]")
plt.ylabel("Swing of instantaneous frequency [GHz]")
plt.title(f"Frequencies swings for various slab lengths for b={b}")
plt.show()