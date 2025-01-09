import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def map_v_to_rgb(theta, module, m_max=None):
    """
    Transform orientation and magnitude of velocity into rgb.

    Parameters:
    --------
    theta: array_like
        Orietation of velocity field.
    module: array_like
        Magnitude of velocity field.
    m_max: float, optional
        Max magnitude to show.

    Returns:
    --------
    RGB: array_like
        RGB corresponding to velocity fields.
    """
    H = theta / 360
    V = module
    if m_max is not None:
        V[V > m_max] = m_max
    else:
        m_max = module.max()
        print(m_max)
    V /= m_max
    S = np.ones_like(H)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    return RGB


def add_colorbar(ax, mmin, mmax, theta_min=0, theta_max=360, orientation="h"):
    """ Add colorbar for the RGB image plotted by plt.imshow() """
    V, H = np.mgrid[0:1:50j, 0:1:180j]
    if orientation == "v":
        V = V.T
        H = H.T
        box = [mmin, mmax, theta_min, theta_max]
    else:
        box = [theta_min, theta_max, mmin, mmax]
    S = np.ones_like(V)
    HSV = np.dstack((H, S, V))
    RGB = hsv_to_rgb(HSV)
    ax.imshow(RGB, origin='lower', extent=box, aspect='auto')
    theta_ticks = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    if orientation == "h":
        ax.set_xticks(theta_ticks)
        ax.set_xticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'module $|{\bf m}|$', fontsize="large")
        ax.set_xlabel("orientation", fontsize="large")
    else:
        ax.yaxis.set_label_position('right')
        ax.yaxis.set_ticks_position("right")
        ax.set_yticks(theta_ticks)
        ax.set_yticklabels([r"$%d\degree$" % i for i in theta_ticks])
        ax.set_ylabel(r'orientation $\theta$', fontsize="large")
        # ax.set_xlabel(r"module $|{\bf m}|$", fontsize="large")
        ax.set_title(r"$|{\bf m}|$", fontsize="large")


def get_colobar_extend(vmin, vmax):
    if vmin is None or vmin == 0.:
        if vmax is None:
            ext = "neither"
        else:
            ext = "max"
    else:
        if vmax is None:
            ext = "min"
        else:
            ext = "both"
    return ext


def plot_profile(rho, px, dx, t, axes=None, linestyle="-"):
    rho_x = np.mean(rho, axis=1)
    px_x = np.mean(px, axis=1)
    x = np.arange(rho_x[0].size) * dx + dx / 2
    Lx = x[-1] + dx/2

    if axes is None:
        figsize = (8, 3)
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=figsize, constrained_layout=True)
    else:
        ax1, ax2 = axes

    ax1.plot(x, rho_x[0], label=r"$S=A$", linestyle=linestyle)
    ax1.plot(x, rho_x[1], label=r"$S=B$", linestyle=linestyle)
    ax2.plot(x, px_x[0], label=r"$S=A$", linestyle=linestyle)
    ax2.plot(x, px_x[1], label=r"$S=B$", linestyle=linestyle)
    ax1.set_xlim(0, Lx)

    ax2.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$\langle\rho_{S}(\mathbf{x}, t) \rangle_y $")
    ax2.set_ylabel(r"$\langle p_{x,S}(\mathbf{x}, t) \rangle_y $")
    if axes is None:
        ax1.legend()
        ax2.legend()

        plt.suptitle(r"$t=%g$" % t)
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



def compare_profiles():
    # i_frame = 64
    i_frame = 108

    fins = [r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2100.npz",  # full nonlinearity
            r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2101.npz"   # full nonlinearity, de-aliased
            ]
    dx = 0.1
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, constrained_layout=True)
    linestyles = ["-", ":"]

    for i, fin in enumerate(fins):
        with np.load(fin, "r") as data:
            t_arr = data["t_arr"]
            rho_arr = data["rho_arr"]
            px_arr = data["px_arr"]
            py_arr = data["py_arr"]
            plot_profile(rho_arr[i_frame], px_arr[i_frame], dx, t_arr[i_frame], axes=axes, linestyle=linestyles[i])
    axes[0].legend()
    axes[1].legend()

    plt.suptitle(r"$t=%g$" % t_arr[i_frame])
    plt.show()
    plt.close()


def show_rho_min_varied_1st_schemes():
    fins = [r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2104.npz",  # RK4, full nonlinearity
            # r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2102.npz",  # partial nonlinearity
            r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2100.npz",  # full nonlinearity
            r"data/L12.8_1.6_Dr0.100_Dt0.01_e0.000_0.000_J1.000_-1.000_dx0.1_h0.0005_s2101.npz"   # full nonlinearity, de-aliased
            ]
    labels = ["RK4, full nonlinearity", "Euler, full nonlinearity", "Euler, full nonlinearity, de-aliased"]

    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    for i, fin in enumerate(fins):
        with np.load(fin, "r") as data:
            t_arr = data["t_arr"]
            rho_arr = data["rho_arr"]
            px_arr = data["px_arr"]
            py_arr = data["py_arr"]
            rho_min = np.array([i.min() for i in rho_arr])
            ax.plot(t_arr, rho_min, label=labels[i])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"${\rm min}(\rho)$")
    ax.set_xlim(0)
    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.legend()
    plt.show()
    plt.close()

def plot_momentum(vx, vy, ax, Lx, Ly, vmax=None):
    theta = np.arctan2(vy, vx)
    theta[theta < 0] += np.pi * 2
    theta *= 180 / np.pi
    module = np.sqrt(vx**2 + vy**2)

    box = [0, Lx, 0, Ly]

    RGB = map_v_to_rgb(theta, module, m_max=vmax)
    ax.imshow(RGB, extent=box, origin="lower")


if __name__ == "__main__":
    dx = 0.4
    dt = 1e-2

    Lx = 51.2
    Ly = 51.2

    Dr = 0.1
    Dt = 0.01

    eta_AA = eta_BB = 0
    eta_AB = 0.75
    eta_BA = -eta_AB
    seed = 1000

    folder = "data"
    L = 51.2
    p_max = 0.25
    # fnpz = f"{folder}/L{Lx:g}_{Ly:g}_Dr{Dr:.3f}_Dt{Dt:g}_e{eta_AA:.3f}_{eta_BB:.3f}_J{eta_AB:.3f}_{eta_BA:.3f}_dx{dx:g}_h{dt:g}_s{seed}.npz"
    fnpz = f"data/L{L:g}_{L:g}_Dr0.100_Dt0.01_e0.000_0.000_J0.750_-0.750_dx0.2_h0.0025_s102.npz"
    with np.load(fnpz, "r") as data:
        t_arr = data["t_arr"]
        rho_arr = data["rho_arr"]
        px_arr = data["px_arr"]
        py_arr = data["py_arr"]
        print(t_arr.size)

        # print(px_arr.shape)
        i_frame = -100
        px_A = px_arr[i_frame, 0]
        py_A = py_arr[i_frame, 0]
        rho_A = rho_arr[i_frame, 0]
        px_B = px_arr[i_frame, 1]
        py_B = py_arr[i_frame, 1]
        rho_B = rho_arr[i_frame, 1]

        theta_A = np.arctan2(py_A, px_A)

        fig, axes = plt.subplots(2, 2, figsize=(8, 7), sharex=True, sharey=True)

        rmin = 0.8
        rmax = 1.6
        axes[0, 0].imshow(rho_A, origin="lower", extent=[0, L, 0, L], vmin=rmin, vmax=rmax)
        plot_momentum(px_A, py_A, axes[1, 0], L, L, p_max)
        im = axes[0, 1].imshow(rho_B, origin="lower", extent=[0, L, 0, L], vmin=rmin, vmax=rmax)
        plot_momentum(px_B, py_B, axes[1, 1], L, L, p_max)

        axes[0, 0].set_title(r"$\phi_A$", fontsize="xx-large")
        axes[0, 1].set_title(r"$\phi_B$", fontsize="xx-large")
        axes[1, 0].set_title(r"$\mathbf{p}_A$", fontsize="xx-large")
        axes[1, 1].set_title(r"$\mathbf{p}_B$", fontsize="xx-large")

        plt.tight_layout(rect=[-0.03, -0.02, 1.02, 1])
        bbox1 = axes[0, 1].get_position().get_points().flatten()
        bbox2 = axes[1, 1].get_position().get_points().flatten()
        fig.subplots_adjust(right=0.87)
        bbox1[0], bbox1[2] = 0.86, 0.05
        bbox1[3] = bbox1[3] - bbox1[1]
        bbox2[0], bbox2[2] = 0.86, 0.08
        bbox2[3] = bbox2[3] - bbox2[1]
        cb_ax1 = fig.add_axes(bbox1)
        cb_ax2 = fig.add_axes(bbox2)

        cb1 = fig.colorbar(im,
                           ax=axes[0, 1],
                           cax=cb_ax1,
                           orientation="vertical",
                           extend="both")
        add_colorbar(cb_ax2, 0, p_max, 0, 360, "v")

        # fig.colorbar(im2, ax=ax2)

        # plt.show()

        plt.savefig("data/cross_sea_pde.pdf")
        plt.close()
       