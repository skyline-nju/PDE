#include <iostream>
#include "solver.h"
#include "rand.h"
#include "BGL.h"
#include "disorder.h"
#define RT

int main(int argc, char* argv[]) {
  double eta = atof(argv[1]);
  double zeta = atof(argv[2]);
  double dt = atof(argv[3]);
  double Lx = atof(argv[4]);
  double Ly = Lx;
  int Nx = atoi(argv[5]);
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
  std::cout << "noise = " << eta << "\n";
  std::cout << "disorder = " << zeta << "\n";
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
  snprintf(basename, 100, "BGL_eta%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, zeta, rho0, Lx, Ly, Nx, Ny, dt);

  char order_para_filename[100];
  snprintf(order_para_filename, 100, "../data/%s.dat", basename);

  char snap_filename[100];
  snprintf(snap_filename, 100, "../data/%s.bin", basename);

  BGL_Solver solver(Nx, Ny, Lx, Ly, NFILED, dt, eta, rho0, D0, do_antialiasing);

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
