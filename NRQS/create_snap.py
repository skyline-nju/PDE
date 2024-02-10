import numpy as np
import os


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


def duplicate(fin, nx, ny, i_frame=-1):
    path = os.path.dirname(fin)
    basename = os.path.basename(fin)

    Lx = float(basename.lstrip("L").split("_")[0])
    Ly = float(basename.lstrip("L").split("_")[1])
    Lx_new = Lx * nx
    Ly_new = Ly * ny

    basename_new = basename.replace("L%g_%g" % (Lx, Ly), "L%g_%g" % (Lx_new, Ly_new))
    fout = f"{path}/{basename_new}"

    with np.load(fin, "r") as data:
        rho = data["rho_arr"][i_frame]
        px = data["px_arr"][i_frame]
        py = data["py_arr"][i_frame]
    
    rho_new = np.tile(rho, (1, ny, nx))
    px_new = np.tile(px, (1, ny, nx))
    py_new = np.tile(py, (1, ny, nx))
    t_new = np.array([0.])

    np.savez_compressed(fout, t_arr=t_new, rho_arr=rho_new, px_arr=px_new, py_arr=py_new)