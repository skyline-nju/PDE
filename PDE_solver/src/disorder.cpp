#include "disorder.h"
#include <iostream>

void load_random_torques(double zeta, double *rand_torques,
                         int Nx, int Ny, int Mx, int My) {
  char fname[100];
  snprintf(fname, 100, "../data/disorder_realization/RT_Nx%d_Ny%d_Lx%d_Ly%d.bin",
           Nx, Ny, Mx, My);
  int N = Nx * Ny;
  std::ifstream fin(fname, std::ios::binary);
  fin.read((char *)rand_torques, sizeof(double) * N);
  for (int i = 0; i < N; i++) {
    rand_torques[i] *= zeta;
  }
  fin.close();
  std::cout << "load the quenched disorder from " << fname << std::endl;
}

void load_random_fields(double zeta, double *RFx, double *RFy,
                        int Nx, int Ny, int Mx, int My) {
  int N = Nx * Ny;
  double *rand_torques = new double[N];
  load_random_torques(1., rand_torques, Nx, Ny, Mx, My);
  for (int i = 0; i < N; i++) {
    RFx[i] = zeta * cos(rand_torques[i]);
    RFy[i] = zeta * sin(rand_torques[i]);
  }
  delete[] rand_torques;

  double vx_m = 0;
  double vy_m = 0;
  for (int i = 0; i < N; i++) {
    vx_m += RFx[i];
    vy_m += RFy[i];
  }
  vx_m /= N;
  vy_m /= N;
  for (int i = 0; i < N; i++) {
    RFx[i] -= vx_m;
    RFy[i] -= vy_m;
  }

  vx_m = vy_m = 0;
  for (int i = 0; i < N; i++) {
    vx_m += RFx[i];
    vy_m += RFy[i];
  }
  std::cout << "vx_m = " << vx_m / N << "\n";
  std::cout << "vy_m = " << vy_m / N << std::endl;
}
