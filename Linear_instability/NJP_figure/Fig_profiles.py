import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
from add_line import add_line


def get_q1(q_radial, Sq):
    """ Get the first moment of s(q, t). """
    I0 = np.sum(Sq)
    I1 = np.sum(q_radial * Sq)
    q1 = I1 / I0
    return q1



def plot_profile_R_order_para():
    fig = plt.figure(figsize=(12, 4), layout="constrained")
    subfigs = fig.subfigures(1, 3, wspace=0.0001, hspace=0.0001, width_ratios=[1, 1, 1])
    ax_left = subfigs[0].subplots(3, 1, sharex=True)
    ax1, ax2, ax3 = ax_left
    ax4 = subfigs[1].subplots(1, 1)
    ax5 = subfigs[2].subplots(1, 1)

    with np.load("data/time_ave_profile/LA_CCB_profiles_p18_14.5_L480_160.npz", "rb") as data:
        x, fx = data["x"], data["fx"]
        x /= 480
        ax1.plot(x, fx[0] / 10, label=r"$S=A$", c="tab:blue")
        ax1.plot(x, fx[1] / 10, label=r"$S=B$", c="tab:red")
    
    with np.load("data/instant_profile/L480_160_Dr0.100_k0.70_p2.25_7.35_r5_5_5_e-2.000_J0.500_-0.500_1001.npz", "rb") as data:
        t, x, fields = data["t"], data["x"], data["fields"]
        ax2.plot(x/480, fields[0, 0]/5, c="tab:blue")
        ax2.plot(x/480, fields[0, 1]/5, c="tab:red")

    with np.load("data/instant_profile/L120_40_Dr0.100_k0.70_p10_28_r10_10_10_e-2.000_J0.500_-0.500_h0.100_1000.npz", "rb") as data:
        t, x, fields = data["t"], data["x"], data["fields"]
        ax3.plot(x/120, fields[-1, 0]/10, c="tab:blue")
        ax3.plot(x/120, fields[-1, 1]/10, c="tab:red")

    files_sq = ["data/Sq/L1280_1280_Dr0.100_k0.70_p3_6_r10_10_10_e-2.000_J0.500_-0.500_h0.333_1000.npz",
                "data/Sq/L1280_1280_Dr0.100_k0.70_p5.5_3.85_r10_10_10_e-2.000_J0.500_-0.500_h0.333_1000.npz",
                "data/Sq/L640_640_Dr0.100_k0.70_p4.5_14.7_r10_10_10_e-2.000_J0.500_-0.500_h0.333_1000.npz",
                "data/Sq/L640_640_Dr0.100_k0.70_p10_28_r10_10_10_e-2.000_J0.500_-0.500_h0.333_1000.npz",
                "data/Sq/L640_640_Dr0.100_k0.70_p17_22.5_r10_10_10_e-2.000_J0.500_-0.500_h0.333_1000.npz"
                ]
    labels = ["G+LB", "G+CCB", "G+LB+\nCCB", "LB+CCB", "CCB"]
    mks =["o", "s", "v", "p", "D"]
    for j, fin in enumerate(files_sq):
        with np.load(fin, "r") as data:
            t, q, rho_Sqt, rho_var = data["t"], data["q"], data["rho_Sqt"], data["rho_var"]
        q1_A, q1_B = np.zeros((2, t.size))
        for i in range(t.size):
            # q1_A[i] = get_q1(q, rho_Sqt[0][i])
            q1_B[i] = get_q1(q, rho_Sqt[1][i])
        if j == 1:
            mask = t < 2.5e6
            t = t[mask]
            q1_B = q1_B[mask]
        ax4.plot(t, 2*np.pi/q1_B, mks[j], fillstyle="none", label=labels[j], ms=5)
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.legend(loc=(0.03, 0.6), fontsize="large", borderpad=0.05, frameon=True, handletextpad=0.2, handlelength=1)

    L_arr = [40, 80, 160, 320, 640]
    jm1 = [0.02447, 0.01205, 0.00574, 0.00267]
    jm2 = [0.05016, 0.02543, 0.01392, 0.007394, 0.003154]
    jm3 = [0.54276, 0.6023, 0.098, 0.04883, 0.0222]

    ax5.plot(L_arr, jm3, "v", label="G+LB+CCB", fillstyle="none", c="tab:green", ms=5)
    ax5.plot(L_arr, jm2, "p", label="LB+CCB", fillstyle="none", c="tab:red", ms=5)
    ax5.plot(L_arr[:-1], jm1, "D", label="CCB", fillstyle="none", c="tab:purple", ms=5)

    ax5.set_xscale("log")
    ax5.set_yscale("log")

    ax1.set_xlim(0, 1)
    ax1.set_ylabel(r"$\langle\rho_S\rangle_{y, t}$", fontsize="x-large")
    ax2.set_ylabel(r"$\langle\rho_S\rangle_{y}$", fontsize="x-large")
    ax3.set_ylabel(r"$\langle\rho_S\rangle_{y}$", fontsize="x-large")
    ax3.set_xlabel(r"$x/L_x$", fontsize="x-large")
    ax1.legend(loc="lower center", ncol=2, fontsize="large", borderpad=0.2, handlelength=1.5)

    ax4.set_xlabel(r"$t$", fontsize="x-large")
    ax4.set_ylabel(r"$\xi_B$", fontsize="x-large")
    ax5.set_ylabel(r"$\langle|\mathbf{J}|\rangle_t$", fontsize="x-large")
    ax5.set_xlabel(r"$L$", fontsize="x-large")
    ax5.legend(fontsize="large", loc=(0.45, 0.78), borderpad=0.2, handlelength=1)
    # ax5.set_ylim(ymax=0.07)

    label_font_size = "x-large"
    ax1.text(0.91, 0.78, "(a)", fontsize=label_font_size, transform=ax1.transAxes)
    ax2.text(0.91, 0.78, "(b)", fontsize=label_font_size, transform=ax2.transAxes)
    ax3.text(0.91, 0.78, "(c)", fontsize=label_font_size, transform=ax3.transAxes)
    ax4.text(0.91, 0.94, "(d)", fontsize=label_font_size, transform=ax4.transAxes)
    ax5.text(0.91, 0.94, "(e)", fontsize=label_font_size, transform=ax5.transAxes)

    ax2.arrow(0.78, 0.5, 0.08, 0, transform=ax2.transAxes, width=0.04, color="k", ec="k", head_length=0.04)
    ax3.arrow(0.2, 0.8, 0.08, 0, transform=ax3.transAxes, width=0.04, color="k", ec="k", head_length=0.04)
    ax3.arrow(0.65, 0.8, 0.08, 0, transform=ax3.transAxes, width=0.04, color="k", ec="k", head_length=0.04)

    add_line(ax4, 0.35, 0.54, 0.98, 0.13, label=r"$0.13$", c="tab:orange", xl=0.8, yl=0.65)

    add_line(ax4, 0.3, 0.26, 0.95, 1/3, label=r"$0.33$", c="tab:blue", xl=0.55, yl=0.45)
    # add_line(ax4, 0.35, 0.7, 0.65, 1/3, label=None, c="tab:blue")
    add_line(ax4, 0.25, 0.4, 0.65, 1, label=r"$1$", c="tab:green", xl=0.35, yl=0.88)

    add_line(ax5, 0.01, 0.8, 0.95, -1, label=r"$-1$", xl=0.3, yl=0.5)
    add_line(ax5, 0.01, 0.6, 0.95, -1)
    add_line(ax5, 0.01, 0.4, 0.95, -1)

    plt.show()
    # plt.savefig("fig/profile_R_J.pdf")
    plt.close()
 


def plot_rho_v():
    fins = ["data/time_ave_profile/G_LB_profiles_p5_20_r20_L120_40.npz",
            "data/time_ave_profile/G_CCB_profiles_p7.6525_10.5675_L960_160.npz"]
    
    fig, axes = plt.subplots(3, 2, figsize=(8, 5), sharex="col", sharey="row", constrained_layout=True)
    rho0_arr = [20, 10]
    for col, fin in enumerate(fins):
        with np.load(fin, "rb") as data:
            x, fx = data["x"], data["fx"]
        axes[0, col].plot(x, fx[0] / rho0_arr[col], label=r"$S=A$", c="tab:blue")
        axes[0, col].plot(x, fx[1] / rho0_arr[col], label=r"$S=B$", c="tab:red")
        axes[1, col].plot(x, fx[2]/fx[0], label=r"$S=A$", c="tab:blue")
        axes[1, col].plot(x, fx[3]/fx[1], label=r"$S=B$", c="tab:red")
        axes[2, col].plot(x, fx[8] / rho0_arr[col], label=r"$S=A$", c="tab:blue")
        axes[2, col].plot(x, fx[9] / rho0_arr[col], label=r"$S=B$", c="tab:red")

    axes[0, 0].set_xlim(0, 120)
    axes[0, 1].set_xlim(0, 960)
    axes[2, 0].set_ylim(ymax=1.1)


    axes[2, 0].set_xlabel(r"$x$", fontsize="x-large")
    axes[2, 1].set_xlabel(r"$x$", fontsize="x-large")

    # axes[0, 0].set_ylabel(r"$\frac{\langle\rho_S\rangle_{y,t}}{\rho_0}$", fontsize="xx-large")
    axes[0, 0].set_ylabel(r"$\langle\rho_S\rangle_{y,t}\slash\rho_0$", fontsize="x-large")

    # axes[1, 0].set_ylabel(r"$\frac{\langle p_{x,S}\rangle_{y,t}}{\langle\rho_S\rangle_{y,t}}$", fontsize="xx-large")
    axes[1, 0].set_ylabel(r"$\langle p_{x,S}\rangle_{y,t}/\langle\rho_S\rangle_{y,t}$", fontsize="x-large")

    # axes[2, 0].set_ylabel(r"$\frac{\langle \rho_S v_S\rangle_{y,t}}{\rho_0}$", fontsize="xx-large")
    axes[2, 0].set_ylabel(r"$\langle \rho_S v_S\rangle_{y,t}\slash\rho_0$", fontsize="x-large")


    phi1 = [0.2867309, 0.49119514, 0.1376073, 2.55688951]
    axes[0, 0].axhline(phi1[0], linestyle=":", c="tab:blue")
    axes[0, 0].axhline(phi1[1], linestyle=":", c="tab:red")
    axes[0, 0].axhline(phi1[2], linestyle="--", c="tab:blue")
    axes[0, 0].axhline(phi1[3], linestyle="--", c="tab:red")

    axes[0, 0].legend(loc='center right', fontsize="large")
    axes[0, 0].text(0.01, 0.8, "(a)", fontsize="x-large", transform=axes[0, 0].transAxes)
    axes[0, 1].text(0.01, 0.8, "(b)", fontsize="x-large", transform=axes[0, 1].transAxes)
    axes[1, 0].text(0.01, 0.85, "(c)", fontsize="x-large", transform=axes[1, 0].transAxes)
    axes[1, 1].text(0.01, 0.85, "(d)", fontsize="x-large", transform=axes[1, 1].transAxes)
    axes[2, 0].text(0.01, 0.85, "(e)", fontsize="x-large", transform=axes[2, 0].transAxes)
    axes[2, 1].text(0.01, 0.85, "(f)", fontsize="x-large", transform=axes[2, 1].transAxes)
    plt.show()
    # plt.savefig("fig/Fig_profile_rho_v.pdf")
    plt.close()
           

if __name__ == "__main__":
    plot_profile_R_order_para()
    