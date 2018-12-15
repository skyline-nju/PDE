#include <iostream>
#include "solver.h"
#include "rand.h"
#include "BGL.h"
#include "disorder.h"

int main(int argc, char* argv[]) {
  double Lx = 64;
  double Ly = 64;
  int Nx = 128;
  int Ny = 128;

  double dt = 0.01;
  double D0 = 0.5;
  double eta = 0.4;
  double rho0 = 1.;
  double zeta = 0.02;
  int do_antialiasing = 1;
  double noise_ini_cond = 0.001;

  int seed = 2;
  Ran myran(seed);

  double *RFx = new double[Nx * Ny];
  double *RFy = new double[Nx * Ny]; 
  ini_rand_field(zeta, RFx, RFy, Nx, Ny, int(Lx), int(Ly), myran);

  char basename[100];
  snprintf(basename, 100, "BGL_eta%g_zeta%g_r%g_Lx%g_Ly%g_Nx%d_Ny%d_dt%g",
    eta, zeta, rho0, Lx, Ly, Nx, Ny, dt);

  char order_para_filename[100];
  snprintf(order_para_filename, 100, "../data/%s.dat", basename);

  char snap_filename[100];
  snprintf(snap_filename, 100, "../data/%s.bin", basename);

  int n_save_order_para = 100;
  int n_save_bin = 1000;
  int n_steps = 2000;

  BGL_Solver solver(Nx, Ny, Lx, Ly, NFILED, dt, eta, rho0, D0, do_antialiasing);

  solver.ini_fields(noise_ini_cond, myran, do_antialiasing);

  solver.save_order_para(0, order_para_filename, 0);
  solver.save_fields(0, snap_filename, 0);

  for (int i = 1; i <= n_steps; i++) {
    solver.one_step(dt, RFx, RFy);
    double t = dt * i;
    if (i % n_save_order_para == 0) {
      solver.save_phi(t, order_para_filename, 1);
    }
    if (i % n_save_bin == 0) {
      solver.save_snap(t, snap_filename, 1);
    }
  }
}
