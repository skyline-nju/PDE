/**
 * @file disorder.h
 * @author Yu Duan (duanyu100@yeah.net)
 * @brief 
 * @version 0.1
 * @date 2018-12-15
 * 
 * initialize the quenched disorder
 * @copyright Copyright (c) 2018
 * 
 */

#pragma once
#include <cmath>
#include "rand.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

template <typename TRan>
void ini_rand_torque(double zeta, double *torques, int N, TRan &myran) {
  double sep = 1. / (N - 1);
  for (int i = 0; i < N; i++) {
    torques[i] = (i * sep - 0.5) * zeta * M_PI * 2.;
  }
  shuffle(torques, N, myran);
}

template <typename TRan>
void ini_rand_torque(double zeta, double *torques, int Nx, int Ny, int Mx, int My, TRan &myran) {
  int N = Nx * Ny;
  int M = Mx * My;
  double *tau_coarse = new double[M];
  ini_rand_torque(zeta, tau_coarse, M, myran);

  int bx = Nx / Mx;
  int by = Ny / My;
  for (int x = 0; x < Mx; x++) {
    for (int y = 0; y < My; y++) {
      int pos_coarse = y + x * My;
      for (int i = 0; i < bx; i++) {
        for (int j = 0; j < by; j++) {
          int pos_fine = j + y * by + (i + x * bx) * Ny;
          torques[pos_fine] = tau_coarse[pos_coarse];
        }
      }
    }
  }

  delete[] tau_coarse;
}

template <typename TRan>
void ini_rand_field(double zeta, double *RFx, double *RFy, int Nx, int Ny, int Mx, int My, TRan &myran) {
  int N = Nx * Ny;
  double *torques = new double[N];
  ini_rand_torque(1., torques, Nx, Ny, Mx, My, myran);
  for (int i = 0; i < N; i++) {
    RFx[i] = zeta * cos(torques[i]);
    RFy[i] = zeta * sin(torques[i]);
  }
  delete[] torques;
}

