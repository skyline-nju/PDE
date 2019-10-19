#include <iostream>
#include "solver.h"
#include "rand.h"
#include "BGL.h"
#include "disorder.h"
// #define RF

int main(int argc, char* argv[]) {
  double dt = atof(argv[1]);
  double Lx = atof(argv[2]);
  int Nx = atoi(argv[3]);
  double eta = atof(argv[4]);
  double zeta = atof(argv[5]);
  std::string disorder_t = "RP";

  double Ly = Lx;
  int Ny = Nx;
  double rho0 = 1.;
  double D0 = atof(argv[6]);
  int do_antialiasing = 1;
  double noise_ini_cond = 0.001;
  int seed = 1;

  int n_save_order_para = 200;
  int n_save_bin = 200;
  int n_steps = 200000;

  std::cout << "--------Parameters--------\n";
  std::cout << "noise = " << eta << "\n";
  std::cout << "disorder = " << zeta << "\n";
  std::cout << "rho0 = " << rho0 << "\n";
  std::cout << "dt = " << dt << std::endl;

  Ran myran(seed);

  char basename[100];
  snprintf(basename, 100, "%s_eta%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g_D%g",
    disorder_t.c_str(), eta, zeta, rho0, Lx, Ly, Nx, Ny, dt, D0);

  char order_para_filename[100];
  snprintf(order_para_filename, 100, "../data/%s.dat", basename);

  char snap_filename[100];
  snprintf(snap_filename, 100, "../data/%s.bin", basename);

  BGLSolverBase* solver = nullptr;
  if (zeta > 0) {
    if (disorder_t == "RF") {
      solver = new BGL_RF(Nx, Ny, Lx, Ly, NFILED, eta, zeta, rho0, D0, do_antialiasing);
    } else if (disorder_t == "RP") {
      solver = new BGL_RP(Nx, Ny, Lx, Ly, NFILED, eta, zeta, rho0, D0, do_antialiasing);
    } else {
      std::cout << "the type of disorder is not correct" << std::endl;
      exit(1);
    }
  } else {
    // solver = new BGL_pure(Nx, Ny, Lx, Ly, NFILED, eta, rho0, D0, do_antialiasing);
    solver = new BGL_RP(Nx, Ny, Lx, Ly, NFILED, eta, zeta, rho0, D0, do_antialiasing);
  }

  solver->ini_fields(noise_ini_cond, myran, do_antialiasing);

  solver->save_order_para(0, order_para_filename, 0);

  solver->save_fields(0, snap_filename, 0);

  for (int i = 1; i <= n_steps; i++) {
    solver->one_step(dt);
    double t = dt * i;
    if (i % n_save_order_para == 0) {
      solver->save_phi(t, order_para_filename, 1);
    }
    if (i % n_save_bin == 0) {
      solver->save_snap(t, snap_filename, 1);
    }
  }
}
