#pragma once

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <chrono>

template <typename T>
void save_order_para(const T* f, int n_fields, int N, double t,
                     std::string name, int append,
                     std::chrono::time_point<std::chrono::system_clock> &real_t0) {
  double order[n_fields];
  double vx_normed = 0;
  double vy_normed = 0;
  double *rho_arr = new double[N];
  for (int k = 0; k < n_fields; k++) {
    order[k] = 0.;
    for (int i = 0; i < N; i++) {
      order[k] += f[k + n_fields * i];
      if (k == 1) {
        vx_normed += f[n_fields * i] * f[k + n_fields * i];
      } else if (k == 2) {
        vy_normed += f[n_fields * i] * f[k + n_fields * i];
      } else if (k == 0) {
        rho_arr[i] = f[n_fields * i];
      }
    }
    order[k] /= N;
  }
  double rho_min = *std::min_element(rho_arr, rho_arr + N);
  delete[] rho_arr;

  vx_normed /= (order[0] * N);
  vy_normed /= (order[0] * N);

  std::ofstream fout;
  if (append) {
    fout.open(name.c_str(), std::ios::out|std::ios::app);
  } else {
    fout.open(name.c_str(), std::ios::out);
  }
  fout << t << "\t";
  for (int k = 0; k < n_fields; k++) {
    fout << order[k] << "\t";
  }
  fout << vx_normed << "\t" << vy_normed << "\t" << rho_min << "\t";

  auto t_now = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = t_now - real_t0;
  const auto dt = elapsed_time.count();
  const auto hour = int(dt / 3600);
  const auto min = int((dt - hour * 3600) / 60);
  const int sec = dt - hour * 3600 - min * 60;
  fout << hour << ":" << min << ":" << sec << std::endl;

  std::cout << t << "\t" << order[0] << "\t";
  std::cout << sqrt(order[1] * order[1] + order[2] * order[2]) << "\t";
  std::cout << atan2(order[2], order[1]) << "\t" << rho_min << "\t";
  std::cout << hour << ":" << min << ":" << sec << std::endl;

  fout.close();
}

template <typename T>
void save_fields(const T* f, int n_fields, int Nx, int Ny,
                 double t, std::string name, int append) {
  int N = Nx * Ny;
  float *buf = new float[N * n_fields + 1];
  buf[0] = t;
  for (int i_field = 0; i_field < n_fields; i_field++) {
    for (int x=0; x < Nx; x++) {
      for (int y = 0; y < Ny; y++) {
        int pos = y + x * Ny + i_field * N;
        buf[1 + pos] = f[i_field + n_fields * (y + x * Ny)];
      }
    }
  }

  std::ofstream fout;
  if (append) {
    fout.open(name.c_str(), std::ios::binary|std::ios::app);
  } else {
    fout.open(name.c_str(), std::ios::binary);
  }
  fout.write((char*)buf, (N * n_fields + 1) * sizeof(float));
  fout.close();
  delete[] buf;
}
