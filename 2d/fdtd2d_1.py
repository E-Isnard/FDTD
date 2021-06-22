import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
2D FDTD with time and space depandant electric permitivity

We use TMz, which means that E has only one component along z and 
that H has components along x and y. They both depend only on x and y.

We use a normalized version of E: Ẽ_z = √(ɛ/µ)*E
"""

# Units
t_unit = 1E-9  # ns
xy_unit = 1E-3  # mm

xymax = 100E-3/xy_unit
Tmax = 10E-9/t_unit
dxy = 1E-4/xy_unit
# courant_number = c*dt/dx
courant_number = 1/2

# EM constants
eps_0_SI = 8.85418782E-12
mu_0_SI = 4*np.pi*1E-7
eps_0 = 1
mu_0 = 1
c = (eps_0*mu_0)**(-1/2)
dt = courant_number*dxy/c

x = np.arange(0, xymax, dxy)
y = np.arange(0, xymax, dxy)
t = np.arange(0, Tmax, dt)


nxy = len(x)
nt = len(t)
print(f"{nt = }")

Hx = np.zeros((nt, nxy, nxy))
Hy = np.zeros((nt, nxy, nxy))
Ez = np.zeros((nt, nxy, nxy))
mid = int(nxy/2)
print(f"{mid = }")


for i in range(nt-1):
    Hx[i+1, :, 1:] = Hx[i, :, 1:]-courant_number*(Ez[i, :, 1:]-Ez[i, :, :-1])
    Hx[i+1, 1:, :] = Hx[i, 1:, :]+courant_number*(Ez[i, 1:, :]-Ez[i, :-1, :])
    Ez[i+1, :-1, :-1] = Ez[i, :-1, :-1]+courant_number * \
        (Hy[i+1, 1:, :-1]-Hy[i+1, :-1, :-1]-Hx[i+1, :-1, 1:]+Hx[i+1, :-1, :-1])
    
   
    Hx[i+1,mid,mid]=np.sin(2*np.pi*t[i])

    k = int((i+1)/(nt-1)*20)
    print(f'\rProgression:[{k*"#"}{(20-k)*" "}] [{i+1}/{nt-1}]', end='', flush=True)

X, Y = np.meshgrid(x, y)
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(X, Y, Ez[0], cmap="viridis")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_zlim(-10, 10)


def anim(i):
    ax.clear()
    plot = ax.plot_surface(X, Y, Ez[i])
    ax.set_zlim(-10, 10)
    return plot,


a = animation.FuncAnimation(
    fig, anim, interval=1000/10, frames=1000, blit=False, repeat=False)
plt.show()
