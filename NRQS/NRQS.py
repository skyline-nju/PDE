import numpy as np
import matplotlib.pyplot as plt



def plot_profile(rho, px, dx, t):
    rho_x = np.mean(rho, axis=1)
    px_x = np.mean(px, axis=1)
    x = np.arange(rho_x[0].size) * dx + dx / 2
    Lx = x[-1] + dx/2

    figsize = (8, 3)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, constrained_layout=True)

    ax1.plot(x, rho_x[0], label=r"$S=A$")
    ax1.plot(x, rho_x[1], label=r"$S=B$")
    ax2.plot(x, px_x[0], label=r"$S=A$")
    ax2.plot(x, px_x[1], label=r"$S=B$")
    ax1.set_xlim(0, Lx)
    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$\langle\rho_{S}(\mathbf{x}, t) \rangle_y $")
    ax2.set_ylabel(r"$\langle p_{x,S}(\mathbf{x}, t) \rangle_y $")

    ax1.legend()
    ax2.legend()

    fig.suptitle(r"$t=%g$" % t)
    plt.show()
    plt.close()


def show_fields(phi, px, py, t, spacing):
    Ny, Nx = phi[0].shape
    if Nx == Ny:
        figsize = (12, 4.5)
    elif Nx == 4 * Ny:
        figsize = (12, 2)
    elif Nx >= 8 * Ny:
        figsize = (12, 1.8)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

    extent = [0, Nx * spacing, 0, Ny * spacing]
    im1 = ax1.imshow(phi[0], origin="lower", extent=extent)
    im2 = ax2.imshow(phi[1], origin="lower", extent=extent)

    if Nx == Ny:
        im3 = ax3.imshow(px[0] **2 + py[0] **2, origin="lower", extent=extent)
        im4 = ax4.imshow(px[1] **2 + py[1] **2, origin="lower", extent=extent)
        cb3_label = r"$|\mathbf{p}_A|^2$"
        cb4_label = r"$|\mathbf{p}_B|^2$"
    else:
        vmin = px[0].min()
        vmax = px[0].max()
        if np.abs(vmin) > np.abs(vmax):
            vmin = -vmax
        else:
            vmax = -vmin
        im3 = ax3.imshow(px[0], origin="lower", extent=extent, cmap="bwr", vmin=vmin, vmax=vmax)
        vmin = px[1].min()
        vmax = px[1].max()
        if np.abs(vmin) > np.abs(vmax):
            vmin = -vmax
        else:
            vmax = -vmin
        im4 = ax4.imshow(px[1], origin="lower", extent=extent, cmap="bwr", vmin=vmin, vmax=vmax)
        cb3_label = r"$\mathbf{p}_{x,A}$"
        cb4_label = r"$\mathbf{p}_{x,B}$"

    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal")
    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")
    cb3 = plt.colorbar(im3, ax=ax3, orientation="horizontal")
    cb4 = plt.colorbar(im4, ax=ax4, orientation="horizontal")

    # cb2.set_label()
    # cb1.set_label(r"$\rho_A$")
    # cb3.set_label(cb3_label)
    # cb4.set_label(cb4_label)

    ax1.set_title(r"$\rho_A$")
    ax2.set_title(r"$\rho_B$")
    ax3.set_title(cb3_label)
    ax4.set_title(cb4_label)

    fig.suptitle(r"$t=%g$" % t)
    plt.show()
    plt.close()


def show_space_time_densities(rho, dx, t_arr):
    rho_x_t = np.mean(rho, axis=2)
    print(rho.shape, rho_x_t.shape)
    x = np.arange(rho_x_t[0, 1].size) * dx + dx / 2
    Lx = x[-1] + dx/2

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True)

    extent = [0, Lx, t_arr[0], t_arr[-1]]
    im1 = ax1.imshow(rho_x_t[:, 0, :], origin="lower", aspect="auto", extent=extent)
    im2 = ax2.imshow(rho_x_t[:, 1, :], origin="lower", aspect="auto", extent=extent)

    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    ax1.set_xlabel(r"$x$")
    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$t$")
    ax1.set_title(r"$\langle \rho_A(\mathbf{r},t)\rangle_y$")
    ax2.set_title(r"$\langle \rho_B(\mathbf{r},t)\rangle_y$")

    plt.show()
    plt.close()


def remove_last_frames(fname, n=1):
    with np.load(fname, "r") as data:
        t_arr = data["t_arr"][:-n]
        rho_arr = data["rho_arr"][:-n]
        px_arr = data["px_arr"][:-n]
        py_arr = data["py_arr"][:-n]

    np.savez_compressed(fname, t_arr=t_arr, rho_arr=rho_arr, px_arr=px_arr, py_arr=py_arr)


def get_one_frame(fin, fout, i_frame=-1):
    with np.load(fin, "r") as data:
        rho_arr = data["rho_arr"][i_frame]
        px_arr = data["px_arr"][i_frame]
        py_arr = data["py_arr"][i_frame]
        t_arr = np.array([0.])
    np.savez_compressed(fout, t_arr=t_arr, rho_arr=rho_arr, px_arr=px_arr, py_arr=py_arr)


if __name__ == "__main__":
    dx = 0.1
    dt = 1e-3

    Lx = 12.8
    Ly = 1.6

    Dr = 1
    Dt = 0.02

    eta_AA = eta_BB = -2
    eta_AB = 0.1
    eta_BA = -eta_AB
    seed = 111

    fnpz = f"data/L{Lx:g}_{Ly:g}_Dr{Dr:.3f}_Dt{Dt:g}_e{eta_AA:.3f}_{eta_BB:.3f}_J{eta_AB:.3f}_{eta_BA:.3f}_dx{dx:g}_h{dt:g}_s{seed}.npz"
    with np.load(fnpz, "r") as data:
        t_arr = data["t_arr"]
        rho_arr = data["rho_arr"]
        px_arr = data["px_arr"]
        py_arr = data["py_arr"]

        i_frame = 89
        show_fields(rho_arr[i_frame], px_arr[i_frame], py_arr[i_frame], t_arr[i_frame], dx)
        plot_profile(rho_arr[i_frame], px_arr[i_frame], dx, t_arr[i_frame])
        
        # for i, t in enumerate(t_arr):
        #     show_fields(rho_arr[i], px_arr[i], py_arr[i], t, 0.05)


        show_space_time_densities(rho_arr, dx, t_arr)

        # plt.plot(t_arr, rho_arr[])

    # fin = r"data\L12.8_1.6_Dr1.000_Dt0.02_e-2.000_-2.000_J0.100_-0.100_dx0.1_h0.001_s223.npz"
    # fout = r"data\L12.8_1.6_Dr1.000_Dt0.02_e-2.000_-2.000_J0.100_-0.100_dx0.1_h0.001_s111.npz"
    # get_one_frame(fin, fout)
