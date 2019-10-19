import numpy as np
import os
import glob


def read_phi(fin, ncut=500):
    with open(fin, "r") as f:
        lines = f.readlines()[ncut:]
        vx, vy = np.zeros((2, len(lines)))
        for i in range(vx.size):
            s = lines[i].split("\t")
            vx[i] = float(s[2])
            vy[i] = float(s[3])
        phi = np.mean(np.sqrt(vx**2 + vy**2))
        return phi


if __name__ == "__main__":
    data_path = r"E:/data/PDE/RP/L64/data"
    disorder_t = "RP"
    Lx = Ly = 64
    Nx = Ny = 128
    dt = 0.02
    D0 = 0.5
    ncut = 250
    basepat = "%s_eta*_zeta*_r1_Lx%d_Ly%d_Nx%d_Ny%d_dt%g_D%g.dat" % (
        disorder_t, Lx, Ly, Nx, Ny, dt, D0)
    pat = "%s%s%s" % (data_path, os.path.sep, basepat)
    print(os.path.sep)
    files = glob.glob(pat)
    n = len(files)
    print("number of input file", n)
    eta_arr, eps_arr, phi_arr = np.zeros((3, n))
    for i, file in enumerate(files):
        basename = os.path.basename(file)
        str_arr = basename.split("_")
        eta_arr[i] = float(str_arr[1].replace("eta", ""))
        eps_arr[i] = float(str_arr[2].replace("zeta", ""))
        phi_arr[i] = read_phi(file, ncut)

    print("n =", n)
    fout = r"../data/order_para/L%d_N%d_dt%g_D%g.dat" % (Lx, Nx, dt, D0)
    with open(fout, "w") as f:
        for i in range(n):
            f.write("%g\t%g\t%.8f\n" % (eta_arr[i], eps_arr[i], phi_arr[i]))
