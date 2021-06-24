import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import perf_counter

"""
2D FDTD with time and space depandant electric permitivity

We use TMz, which means that E has only one component along z and 
that H has components along x and y. They both depend only on x and y.

We use a normalized version of E: Ẽ_z = √(ɛ0/µ0)*E
"""

# Units
t_unit = 1E-9  # ns
xy_unit = 1E-3  # mm

dxy = 0.05
xymax = 100*dxy
# courant_number = c*dt/dx
courant_number = 1/np.sqrt(2)

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = eps_0_SI
mu_0 = mu_0_SI
c = (eps_0*mu_0)**(-1/2)
dt = courant_number*dxy/c
print(f"{dt = }")
Tmax = 150*dt
x = np.arange(0, xymax, dxy)
y = np.arange(0, xymax, dxy)
t = np.arange(0, Tmax, dt)

nxy = len(x)
nt = len(t)
print(f"{nt = }")
print(f"{nxy = }")
print(f"{c*dt/dxy = }")

Hx = np.zeros((nt, nxy, nxy))
Hy = np.zeros((nt, nxy, nxy))
Ez = np.zeros((nt, nxy, nxy))

if Hx.nbytes / 1024**3 > 1:
    raise MemoryError("Too much memory")

X, Y = np.meshgrid(x, y)
mid = int(nxy/2)
t0 = 20
spread = 6
k_abc = (courant_number-1)/(courant_number+1)
for i in range(nt-1):
    # source = np.sin(2*np.pi*t[i]*1e9)
    source = np.exp(-0.5 * ((t0 - i) / spread) ** 2)
    Ez[i+1, 1:, 1:] = Ez[i, 1:, 1:]+courant_number * \
        (Hy[i, 1:, 1:]-Hy[i, :-1, 1:]-Hx[i, 1:, 1:]+Hx[i, 1:, :-1])
    Hx[i+1, :-1, :-1] = Hx[i, :-1, :-1]+courant_number * \
        (Ez[i+1, :-1, :-1]-Ez[i+1, :-1, 1:])
    Hy[i+1, :-1, :-1] = Hy[i, :-1, :-1]+courant_number * \
        (Ez[i+1, 1:, :-1]-Ez[i+1, :-1, :-1])
    Ez[i+1, mid, mid] += source

    Ez[i+1, 0,:] = Ez[i, 1,:]+k_abc*(Ez[i+1, 1,:]-Ez[i, 0,:])
    Ez[i+1, -1,:] = Ez[i, -2,:]+k_abc*(Ez[i+1, -2,:]-Ez[i, -1,:])
    Ez[i+1, :,0] = Ez[i, :,1]+k_abc*(Ez[i+1, :,1]-Ez[i, :,0])
    Ez[i+1, :,-1] = Ez[i, :,-2]+k_abc*(Ez[i+1, :,-2]-Ez[i, :,-1])
    k = int((i+1)/(nt-1)*20)
    print(
        f'\rProgression:[{k*"#"}{(20-k)*" "}] [{round((i+1)/(nt-1)*100,2)} %]', end='', flush=True)
print()


fig = plt.figure(figsize=(30, 30))
ax = plt.axes(projection="3d")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)


def anim(i):
    ax.clear()
    ax.set_zlim(-0.1, 0.1)
    s = perf_counter()
    plot = ax.plot_surface(X, Y, Ez[i])
    print(perf_counter()-s)
    # plt.title(i)
    return plot,


a = animation.FuncAnimation(
    fig, anim, frames=int((nt-1)*1), blit=False, repeat=False)
# a.save("oui2.gif")
plt.show()
print(Ez)
