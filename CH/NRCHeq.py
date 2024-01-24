import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def get_g_A(phi_A, phi_B, cA_1, cA_2, chi, chi_prime, alpha):
    return 2 * (phi_A-cA_1) * (phi_A-cA_2) * (2*phi_A-cA_1-cA_2) + (chi+alpha) * phi_B + 2 * chi_prime * phi_A * phi_B ** 2

def get_g_B(phi_A, phi_B, cB_1, cB_2, chi, chi_prime, alpha):
    return 2 * (phi_B-cB_1) * (phi_B-cB_2) * (2*phi_B-cB_1-cB_2) + (chi-alpha) * phi_A + 2 * chi_prime * phi_B * phi_A ** 2


def show_fields(phiA, phiB, t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True, constrained_layout=True)
    im1 = ax1.imshow(phiA, origin="lower")
    im2 = ax2.imshow(phiB, origin="lower")
    plt.colorbar(im1, ax=ax1)
    plt.colorbar(im2, ax=ax2)
    fig.suptitle(r"$t=%g$" % t)
    plt.show()
    plt.close()


def get_k2(Nx, Ny, spacing):
    ky = np.zeros(Ny)
    kx = np.zeros(Nx // 2 + 1)
    for i in range(Ny):
        if i < Ny // 2:
            ky[i] = i / (Ny * spacing) * np.pi * 2
        else:
            ky[i] = (i - Ny) / (Ny * spacing) * np.pi * 2
    for i in range(Nx // 2 + 1):
        kx[i] = i / (Nx * spacing) * np.pi * 2
    kxx, kyy = np.meshgrid(kx, ky)
    k2 = kxx ** 2 + kyy ** 2
    return k2


class exEulerFS2:
    def __init__(self, dt, dx, Nx, Ny, kappa, cA_1, cA_2, cB_1, cB_2, chi, chi_prime, alpha):
        k2 = get_k2(Nx, Ny, dx)
        self.coeff1 = 1 - dt  * k2 * k2 * kappa
        self.coeff2 = -dt * k2
        self.cA_1 = cA_1
        self.cA_2 = cA_2
        self.cB_1 = cB_1
        self.cB_2 = cB_2
        self.chi = chi
        self.chi_prime = chi_prime
        self.alpha = alpha
    
    def one_step(self, phiA, phiB, phiA_k, phiB_k):
        g_A = get_g_A(phiA, phiB, self.cA_1, self.cA_2, self.chi, self.chi_prime, self.alpha)
        g_B = get_g_B(phiA, phiB, self.cB_1, self.cB_2, self.chi, self.chi_prime, self.alpha)
        g_A_k = np.fft.rfft2(g_A)
        g_B_k = np.fft.rfft2(g_B)
        phiA_k_next = self.coeff1 * phiA_k + self.coeff2 * g_A_k
        phiB_k_next = self.coeff1 * phiB_k + self.coeff2 * g_B_k
        phiA_next = np.fft.irfft2(phiA_k_next)
        phiB_next = np.fft.irfft2(phiB_k_next)
        return phiA_next, phiB_next, phiA_k_next, phiB_k_next


class simFS1:
    def __init__(self, dt, dx, Nx, Ny, kappa, cA_1, cA_2, cB_1, cB_2, chi, chi_prime, alpha):
        k2 = get_k2(Nx, Ny, dx)
        self.coeff3 = 1 / (1 + kappa * dt * k2 * k2)
        self.coeff4 = -dt * k2
        self.cA_1 = cA_1
        self.cA_2 = cA_2
        self.cB_1 = cB_1
        self.cB_2 = cB_2
        self.chi = chi
        self.chi_prime = chi_prime
        self.alpha = alpha


    
    def one_step(self, phiA, phiB, phiA_k, phiB_k):
        g_A = get_g_A(phiA, phiB, self.cA_1, self.cA_2, self.chi, self.chi_prime, self.alpha)
        g_B = get_g_B(phiA, phiB, self.cB_1, self.cB_2, self.chi, self.chi_prime, self.alpha)
        g_A_k = np.fft.rfft2(g_A)
        g_B_k = np.fft.rfft2(g_B)
        phiA_k_next = self.coeff3 * (phiA_k + self.coeff4 * g_A_k)
        phiB_k_next = self.coeff3 * (phiB_k + self.coeff4 * g_B_k)
        phiA_next = np.fft.irfft2(phiA_k_next)
        phiB_next = np.fft.irfft2(phiB_k_next)
        return phiA_next, phiB_next, phiA_k_next, phiB_k_next


if __name__ == "__main__":
    # fig, ax = plt.subplots()
    # t = np.linspace(0, 3, 40)
    # g = -9.81
    # v0 = 12
    # z = g * t**2 / 2 + v0 * t

    # v02 = 5
    # z2 = g * t**2 / 2 + v02 * t

    # scat = ax.scatter(t[0], z[0], c="b", s=5, label=f'v0 = {v0} m/s')
    # line2 = ax.plot(t[0], z2[0], label=f'v0 = {v02} m/s')[0]
    # ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
    # ax.legend()


    # def update(frame):
    #     # for each frame, update the data stored on each artist.
    #     x = t[:frame]
    #     y = z[:frame]
    #     # update the scatter plot:
    #     data = np.stack([x, y]).T
    #     scat.set_offsets(data)
    #     # update the line plot:
    #     line2.set_xdata(t[:frame])
    #     line2.set_ydata(z2[:frame])
    #     return (scat, line2)


    # ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
    # plt.show()
    # plt.close()

    dt = 1e-4
    spacing = 0.05
    Nx = Ny = 128 # Nx and Ny should be even
    Lx = Ly = Nx * spacing

    ## Average concentration
    phiA_0 = 0.35
    # phiB_0 = 0.3
    phiB_0 = 0.42

    ## Nonreciprocity
    alpha = 0.4

    cA_1 = 0.2
    cA_2 = 0.5
    cB_1 = 0.1
    cB_2 = 0.5
    chi = -0.2
    chi_prime = 0
    kappa = 1e-4

    np.random.seed(1234)
    phiA = np.ones((Ny, Nx)) * phiA_0 + (np.random.rand(Ny, Nx) - 0.5) * 0.001
    phiB = np.ones((Ny, Nx)) * phiB_0 + (np.random.rand(Ny, Nx) - 0.5) * 0.001
    phiA_k = np.fft.rfft2(phiA)
    phiB_k = np.fft.rfft2(phiB)

    solver = simFS1(dt, spacing, Nx, Ny, kappa, cA_1, cA_2, cB_1, cB_2, chi, chi_prime, alpha)

    n_step = 200000
    dn_out = 10000

    for i in range(n_step):
        if i % dn_out == 0:
            show_fields(phiA, phiB, i * dt)
        phiA, phiB, phiA_k, phiB_k = solver.one_step(phiA, phiB, phiA_k, phiB_k)
