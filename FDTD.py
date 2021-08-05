import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from matplotlib import animation
from scipy.signal.signaltools import hilbert
from scipy.integrate import simps


def progress(i, n):
    i += 1
    k = int(i/n*20)
    print(
        f'\rProgression:[{k*"#"}{(20-k)*" "}] [{(i/n*100):.0f} %]', end='' if i != n else "\n", flush=True)


class FDTD:
    def __init__(self, L, delta, T_max, d, source_func, source_pos, L_slab, slab_pos, epsr_func, E0_func=lambda *xi: 0, H0_func=lambda *xi: 0, H0_func2=lambda *xi: 0, J_func=lambda T, *Xi: np.zeros(T.shape), cfl=None, boundary_condition="Mur", eps_0=8.85418782e-12, mu_0=4*np.pi*1e-7, memory_limit=1):
        if d != 1 and d != 2:
            raise ValueError("Dimension should be 1 or 2.")
        self.eps_0 = eps_0
        self.mu_0 = mu_0
        self.c = (self.eps_0*self.mu_0)**(-1/2)
        self.d = d
        self.L = L
        self.delta = delta
        self.T_max = T_max
        self.cfl = cfl if cfl is not None else 0.95/np.sqrt(d)
        self.dt = self.cfl*self.delta/self.c
        self.n_space = int(L/delta)+1
        self.nt = int(T_max/self.dt)+1
        E_shape = (self.nt,)+d*(self.n_space,)
        H_shape = (self.nt,)+d*(self.n_space-1,)
        self.Hy = np.zeros(H_shape)
        if self.Hy.nbytes/1024**3 > memory_limit:
            raise MemoryError(
                f"Memory limit overrun ({self.Hy.nbytes/1024**3:.2f} GB > {memory_limit} GB)")
        self.Hx = np.zeros(H_shape) if d != 1 else None
        self.Ez = np.zeros(E_shape)
        if d == 1:
            x = np.linspace(0, L, self.n_space)
            x2 = x[:-1]+self.delta/2
            t = np.linspace(0, T_max, self.nt)
            # t2 = t+self.dt/2
            X, T = np.meshgrid(x, t)
            # X2, T2 = np.meshgrid(x, t2)
            self.Hy[0] = H0_func(x2)
            self.Ez[0] = E0_func(x)
            # self.Hy[0] = self.Hy[0]+self.cfl/2*(self.Ez[0,1:]-self.Ez[0,:-1])
            self.J = J_func(T, X)
        if d == 2:
            x = np.linspace(0, L, self.n_space)
            y = np.linspace(0, L, self.n_space)
            t = np.linspace(0, T_max, self.nt)
            X, Y, T = np.meshgrid(x, y, t, indexing="ij")
            X2, Y2 = np.meshgrid(x, y)
            self.Ez[0] = E0_func(X2, Y2)
            self.Hy[0] = H0_func(X2, Y2)
            self.Hx[0] = H0_func(X2, Y2)
            self.J = J_func(T, X, Y)
        self.source_func = source_func
        self.source_pos = source_pos
        self.L_slab = L_slab
        self.slab_pos = slab_pos
        self.epsr_func = epsr_func
        self.boundary_condition = boundary_condition

    def run(self, progress_bar=True, info=True):
        ca = np.ones(self.d*(self.n_space,))
        cb = self.cfl*np.ones(self.d*(self.n_space,))
        k1 = int(self.slab_pos/self.delta)
        shift = int(self.L_slab/self.delta)
        k2 = k1+shift
        coeff_mur = (self.cfl-1)/(self.cfl+1)
        if self.d == 1:
            ks = int(self.source_pos/self.delta)
            if info:
                start = perf_counter()
            for n in range(self.nt-1):
                tn = n*self.dt
                ca[k1:k2] = self.epsr_func(tn)/self.epsr_func(tn+self.dt)
                cb[k1:k2] = self.cfl/self.epsr_func(tn+self.dt)

                self.Ez[n+1, 1:-1] = ca[1:-1]*self.Ez[n, 1:-1]+cb[1:-1] * \
                    (self.Hy[n, 1:]-self.Hy[n, :-1] -
                     self.J[n+1, 1:-1]*self.delta)

                if self.boundary_condition == "Mur":
                    self.Ez[n+1, 0] = self.Ez[n, 1]+coeff_mur * \
                        (self.Ez[n+1, 1]-self.Ez[n, 0])
                    self.Ez[n+1, -1] = self.Ez[n, -2]+coeff_mur * \
                        (self.Ez[n+1, -2]-self.Ez[n, -1])

                if self.boundary_condition == "PEC":
                    self.Ez[n+1, -1] = 0
                    self.Ez[n+1, 0] = 0

                if self.boundary_condition == "SM":
                    self.Ez[n+1, 0] = self.Hy[n+1, 0]
                    self.Ez[n+1, -1] = -self.Hy[n+1, -1]

                self.Ez[n+1, ks] += self.source_func(tn+self.dt)

                self.Hy[n+1] = self.Hy[n]+self.cfl * \
                    (self.Ez[n+1, 1:]-self.Ez[n+1, :-1])

                if progress_bar:
                    progress(n, self.nt-1)
            if info:
                print(f"Calculations took {(perf_counter()-start):.2f} s")
        if self.d == 2:
            ks1 = int(self.source_pos[0]/self.delta)
            ks2 = int(self.source_pos[1]/self.delta)
            j1 = 0
            j2 = self.n_space-1
            if info:
                start = perf_counter()
            for n in range(self.nt-1):
                tn = n*self.dt
                ca[j1:j2, k1:k2] = self.epsr_func(
                    tn)/self.epsr_func(tn+self.dt)
                cb[j1:j2, k1:k2] = self.cfl/self.epsr_func(tn+self.dt)

                self.Hx[n+1, :, 1:] = self.Hx[n, :, 1:] - \
                    self.cfl*(self.Ez[n, :, 1:]-self.Ez[n, :, :-1])
                self.Hy[n+1, 1:, :] = self.Hy[n, 1:, :] + \
                    self.cfl*(self.Ez[n, 1:, :]-self.Ez[n, :-1, :])
                self.Ez[n+1, :-1, :-1] = ca[:-1, :-1]*self.Ez[n, :-1, :-1]+cb[:-1, :-1]*(
                    self.Hy[n+1, 1:, :-1]-self.Hy[n+1, :-1, :-1]-self.Hx[n+1, :-1, 1:]+self.Hx[n+1, :-1, :-1])

                if self.boundary_condition == "Mur":
                    self.Ez[n+1, 0, :] = self.Ez[n, 1, :]+coeff_mur * \
                        (self.Ez[n+1, 1, :]-self.Ez[n, 0, :])
                    self.Ez[n+1, -1, :] = self.Ez[n, -2, :]+coeff_mur * \
                        (self.Ez[n+1, -2, :]-self.Ez[n, -1, :])
                    self.Ez[n+1, :, 0] = self.Ez[n, :, 1]+coeff_mur * \
                        (self.Ez[n+1, :, 1]-self.Ez[n, :, 0])
                    self.Ez[n+1, :, -1] = self.Ez[n, :, -2]+coeff_mur * \
                        (self.Ez[n+1, :, -2]-self.Ez[n, :, -1])
                if self.boundary_condition == "PEC":
                    self.Ez[n+1, 0, :] = 0
                    self.Ez[n+1, :, 0] = 0
                    self.Ez[n+1, -1, :] = 0
                    self.Ez[n+1, :, -1] = 0

                self.Ez[n+1, ks1, ks2] += self.source_func(tn+self.dt)
                if progress_bar:
                    progress(n, self.nt-1)
            if info:
                print(f"Calculations took {(perf_counter()-start):.2f} s")

    def anim1d(self, y_low, y_high, interval=1e-3, save=False, show=True):
        if self.d != 1:
            raise ValueError("This method must be used only in 1d")
        k1 = int(self.slab_pos/self.delta)
        shift = int(self.L_slab/self.delta)
        k2 = k1+shift
        x = np.linspace(0, self.L, self.n_space)
        x2 = x[:-1]+self.delta/2
        fig = plt.figure()
        line, = plt.plot(x, self.Ez[0])
        line2, = plt.plot(x2, self.Hy[0])
        plt.title("Propagation of $\\tilde{E}_z$ and $H_y$")
        plt.xlabel("x [m]")
        plt.ylabel("Amplitude [A/m]")
        plt.legend(["$\\tilde{E}_z$", "$H_y$"])
        plt.fill_betweenx(y=[y_low, y_high], x1=x[k1],
                          x2=x[k2], color="grey", alpha=0.8)

        plt.ylim(y_low, y_high)
        plt.xlim(0, self.L)

        def animate(i):
            y1 = self.Ez[i].reshape((self.n_space, 1))
            y2 = self.Hy[i].reshape((self.n_space-1, 1))
            line.set_data(x, y1)
            line2.set_data(x2, y2)

        frames = self.nt/5 if save else self.nt
        ani = animation.FuncAnimation(
            fig, animate, interval=interval, frames=self.nt)

        if save:
            ani.save("anim.gif", fps=60)
        if show:
            plt.show()
        plt.close()

    def plotE1d(self, xo, save=False, show=True):
        if self.d != 1:
            raise ValueError("This method must be used only in 1d")
        ko = int(xo/self.delta)
        t = np.arange(0, self.T_max, self.dt)
        plt.plot(t, self.Ez[:, ko])
        plt.title(
            "Normalized Electric Field $\\tilde{E}_z(x_o,t)$")
        plt.xlabel("time [s]")
        plt.ylabel("$\\tilde{E}_z(x_o,t)$ [A/m]")
        plt.grid(ls="--")
        if save:
            plt.savefig("plot.png")
        if show:
            plt.show()
        plt.close()

    def plotE2d(self, xo, yo, save=False, show=True):
        if self.d != 2:
            raise ValueError("This method must be used only in 2d")
        ko1 = int(xo/self.delta)
        ko2 = int(yo/self.delta)
        t = np.arange(0, self.T_max, self.dt)
        plt.plot(t, self.Ez[:, ko1, ko2])
        plt.title(
            "Normalized Electric Field $\\tilde{E}_z(x_o,y_o,t)$")
        plt.xlabel("time [ns]")
        plt.ylabel("$\\tilde{E}_z(x_0,y_o,t)$ [A/m]")
        plt.grid(ls="--")
        if save:
            plt.savefig("plot.png")
        if show:
            plt.show()
        plt.close()

    def animContour(self, vmin=None, vmax=None, interval=1e-3, cmap="jet", save=False):
        if self.d != 2:
            raise ValueError("This method should be used in 2d only")
        x = np.linspace(0, self.L, self.n_space)
        y = np.linspace(0, self.L, self.n_space)
        X, Y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes()
        plt_tmp = ax.contourf(
            X, Y, self.Ez[len(self.Ez)//2], vmin=vmin, vmax=vmax, levels=100, cmap=cmap)
        plt.colorbar(plt_tmp)
        ax.contourf(X, Y, self.Ez[0], vmin=vmin,
                    vmax=vmax, levels=100, cmap=cmap)
        k1 = int(self.slab_pos/self.delta)
        shift = int(self.L_slab/self.delta)
        k2 = k1+shift
        j1 = 0
        j2 = self.n_space-1
        x1 = x[k1]
        x2 = x[k2]
        y1 = y[j1]
        y2 = y[j2]
        coord = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        # repeat the first point to create a 'closed loop'
        coord.append(coord[0])

        xs, ys = zip(*coord)

        def anim2(i):
            ax.clear()
            plt.plot(xs, ys, color="black", lw=3)
            plot = ax.contourf(X, Y, self.Ez[i], vmin=vmin,
                               vmax=vmax, levels=100, cmap=cmap)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            return plot,

        a2 = animation.FuncAnimation(
            fig, anim2, interval=interval, frames=int((self.nt-1)), blit=False, repeat=False)

        if save:
            print("Contour Animation:")
            a2.save("contour.mp4", fps=60, progress_callback=progress)
        plt.show()

    def animSurface(self, z_low, z_high, interval=1e-3, save=False):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection="3d", xlabel="X", ylabel="Y")
        x = np.linspace(0, self.L, self.n_space)
        y = np.linspace(0, self.L, self.n_space)
        X, Y = np.meshgrid(x, y)

        def anim(i):
            ax.clear()
            ax.set_zlim(z_low, z_high)
            plot = ax.plot_surface(X, Y, self.Ez[i])
            return plot,
        a = animation.FuncAnimation(
            fig, anim, interval=interval, frames=self.nt-1, repeat=False)
        if save:
            print("Surface Animation:")
            a.save("SurfaceAnimation.mp4", fps=60, progress_callback=progress)
        plt.show()

    def instant_freq(self, xo):
        ko = int(xo/self.delta)
        phi = np.unwrap(np.angle(hilbert(self.Ez[:, ko])))
        freq = 1/(np.pi*2)*np.gradient(phi, self.dt)
        return freq

    def spectrum(self, xo):
        ko = int(xo/self.delta)
        freqs = np.fft.fftfreq(self.nt, d=self.dt)
        spectrum = np.fft.fft(self.Ez[:, ko])
        return freqs, spectrum

    def H_on_E_grid(self):
        Ho = np.zeros(self.Hy.shape)
        Ho[1:, 1:] = 1/4*(self.Hy[1:, 1:]+self.Hy[1:, :-1] +
                          self.Hy[:-1, 1:]+self.Hy[:-1, :-1])
        return Ho

    def energy(self):
        u_em = np.zeros(self.Hy.shape)
        t = np.linspace(0, self.T_max, self.nt).reshape(-1, 1)
        # u_em[:-1] = self.mu_0/2*(self.epsr_func(t)[:-1]*self.Ez[:-1]**2+((self.Hy[1:]-self.Hy[:-1])/2)**2)
        u_em[:-1] = 1
        u_em[-1] = self.mu_0/2*(self.epsr_func(t[-1])*self.Ez[-1]**2+self.Hy[-1]**2)
        energy = simps(u_em, dx=self.delta)
        return energy
