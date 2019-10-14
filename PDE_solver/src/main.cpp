#include <iostream>
#include "solver.h"
#include "rand.h"
#include "BGL.h"
#include "disorder.h"
// #define RT
#define BINOISE

int main(int argc, char* argv[]) {
  double dt = atof(argv[1]);
  double Lx = atof(argv[2]);
  int Nx = atoi(argv[3]);
  double eta = atof(argv[4]);
#ifdef BINOISE
  double eta_sd = atof(argv[5]);
#ifdef RT
  double zeta = atof(argv[6]);
#endif
#else
#ifdef RT
  double zeta = atof(argv[5]);
#endif
#endif
  double Ly = Lx;
  int Ny = Nx;
  double rho0 = 1.;
  double D0 = 0.5;
  int do_antialiasing = 1;
  double noise_ini_cond = 0.001;
  int seed = 2;

  int n_save_order_para = 200;
  int n_save_bin = 1000;
  int n_steps = 200000;

  std::cout << "--------Parameters--------\n";
#ifdef BINOISE
  std::cout << "collision noise = " << eta << "\n";
  std::cout << "self-diffusion noise = " << eta_sd << "\n";
#else
  std::cout << "noise = " << eta << "\n";
#endif
#ifdef RT
  std::cout << "disorder = " << zeta << "\n";
#endif
  std::cout << "rho0 = " << rho0 << "\n";
  std::cout << "dt = " << dt << std::endl;

  Ran myran(seed);

#ifdef RT
  double *RFx = new double[Nx * Ny];
  double *RFy = new double[Nx * Ny]; 
  load_random_fields(zeta, RFx, RFy, Nx, Ny, int(Lx), int(Ly));
  // ini_rand_field(zeta, RFx, RFy, Nx, Ny, int(Lx), int(Ly), myran);
#endif

  char basename[100];
#ifdef RT
  snprintf(basename, 100, "BGL_eta%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, zeta, rho0, Lx, Ly, Nx, Ny, dt);
#else
#ifdef BINOISE
  snprintf(basename, 100, "BGL_eta%g_etasd%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
           eta, eta_sd, rho0, Lx, Ly, Nx, Ny, dt);
#else
#endif
#endif

#ifdef BINOISE
#ifdef RT
  snprintf(basename, 100, "BGL_eta%g_etasd%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, eta_sd, zeta, rho0, Lx, Ly, Nx, Ny, dt);
#else
  snprintf(basename, 100, "BGL_eta%g_etasd%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, eta_sd, rho0, Lx, Ly, Nx, Ny, dt);
#endif
#else
#ifdef RT
  snprintf(basename, 100, "BGL_eta%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, zeta, rho0, Lx, Ly, Nx, Ny, dt);
#else
  snprintf(basename, 100, "BGL_eta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, rho0, Lx, Ly, Nx, Ny, dt);
#endif
#endif
  char order_para_filename[100];
  snprintf(order_para_filename, 100, "../data/%s.dat", basename);

  char snap_filename[100];
  snprintf(snap_filename, 100, "../data/%s.bin", basename);

#ifdef BINOISE
  BGL_Solver solver(Nx, Ny, Lx, Ly, NFILED, dt, eta, eta_sd, rho0, D0, do_antialiasing);
#else
  BGL_Solver solver(Nx, Ny, Lx, Ly, NFILED, dt, eta, rho0, D0, do_antialiasing);
#endif
  solver.ini_fields(noise_ini_cond, myran, do_antialiasing);

  solver.save_order_para(0, order_para_filename, 0);
  solver.save_fields(0, snap_filename, 0);

  for (int i = 1; i <= n_steps; i++) {
#ifdef RT
    solver.one_step(dt, RFx, RFy);
#else
    solver.one_step(dt);
#endif
    double t = dt * i;
    if (i % n_save_order_para == 0) {
      solver.save_phi(t, order_para_filename, 1);
    }
    if (i % n_save_bin == 0) {
      solver.save_snap(t, snap_filename, 1);
    }
  }
}
