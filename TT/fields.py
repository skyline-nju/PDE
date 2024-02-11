import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def ini_rand(Nx, Ny, rho0, epsilon=0.01):
    delta_rho = (np.random.rand(Ny, Nx) - 0.5) * epsilon
    delta_rho_mean = np.mean(delta_rho)
    rho = rho0 + delta_rho - delta_rho_mean
    mx = (np.random.rand(Ny, Nx) - 0.5) * epsilon
    my = (np.random.rand(Ny, Nx) - 0.5) * epsilon
    t0 = 0.
    return rho, mx, my, t0


def ini_from_file(fname):
    with np.load(fname, "r") as data:
        if data["rho_arr"].ndim == 3:
            rho = data["rho_arr"][-1]
            px = data["mx_arr"][-1]
            py = data["my_arr"][-1]
        elif data["rho_arr"].ndim == 2:
            rho = data["rho_arr"]
            px = data["mx_arr"]
            py = data["my_arr"]
        else:
            print("wrong dimension for input fields")
            sys.exit(1)
        t0 = data["t_arr"][-1]
    return rho, px, py, t0


def ini_fields(fin, mode, spacing, Nx, Ny, rho0, epsilon=0.01):
    if mode == "rand":
        rho, mx, my, t0 = ini_rand(Nx, Ny, rho0, epsilon=epsilon)
        if os.path.exists(fin):
            print("Warning,", fin, "already exists and will be overwritten!")
    elif mode == "resume":
        rho, mx, my, t0 = ini_from_file(fin)
    show_fields(rho, mx, my, t0, spacing)
    return rho, mx, my, t0


def dump_fields(rho_arr, mx_arr, my_arr, t_arr, rho, mx, my, t, i_frame):
    rho_arr[i_frame] = rho
    mx_arr[i_frame] = mx
    my_arr[i_frame] = my
    t_arr[i_frame] = t


def show_fields(phi, px, py, t, spacing):
    Ny, Nx = phi.shape
    if Nx == Ny:
        figsize = (6, 4.5)
    elif Nx == 4 * Ny:
        figsize = (6, 2)
    elif Nx >= 8 * Ny:
        figsize = (6, 1.8)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True, constrained_layout=True)

    extent = [0, Nx * spacing, 0, Ny * spacing]
    im1 = ax1.imshow(phi, origin="lower", extent=extent)

    if Nx == Ny:
        im2 = ax2.imshow(px**2 + py**2, origin="lower", extent=extent)
        cb2_label = r"$|\mathbf{m}|^2$"
    else:
        if px.min() > 0:
            vmin = 0
            vmax = px.max()
        elif px.max() < 0:
            vmin = px.min()
            vmax = 0
        else:
            vmin = px.min()
            vmax = px.max()
        im2 = ax2.imshow(px, origin="lower", extent=extent, vmin=vmin, vmax=vmax)

        cb2_label = r"$\mathbf{m}_{x}$"

    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal")
    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal")

    cb1.set_label(r"$\rho$")
    cb2.set_label(cb2_label)    

    order_para = np.sqrt(np.mean(px) ** 2 + np.mean(py) ** 2) / np.mean(phi)
    fig.suptitle(r"$t=%g, \phi=%.6f, \rho_{\rm min}=%.6f, \rho_{\rm max}=%.6f$" % (t, order_para, np.min(phi), np.max(phi)))
    # print(order_para, np.mean(px), np.mean(py), np.mean(phi))

    plt.show()
    plt.close()


if __name__ == "__main__":
    pass