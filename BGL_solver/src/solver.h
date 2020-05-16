/**
 * @file solver.h
 * @author Yu Duan (duanyu100@yeah.net)
 * @brief 
 * @version 0.1
 * @date 2018-12-13
 * 
 * Integrate the PDEs.
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#pragma once
#include <fftw3.h>
#include <string>
#include <ctime>
#include <chrono>
#include "deriv.h"

class Solver{
public:
  Solver(int Nx, int Ny, double Lx, double Ly, int n_fields);

  ~Solver();

  template <typename TRan>
  void ini_rand(double noise_ini_cond, TRan &myran);

  void ini_from_file(const char *fname, int i_frame);

  void save_order_para(double t, const char *fname, int append) const;

  void save_fields(double t, const char *fname, int append) const;

protected:
  // lattice size
  int Nx_;
  int Ny_;
  int N_; // equal to Nx_ * Ny_

  // number of fields, 3 for d=2: rho, p_x, p_y
  int n_fields_;

  // system size
  double Lx_;
  double Ly_;
  double L_;
  double dx_;
  double dy_;

  // fields
  int alloc_real_;
  double *f_; // with dimensions (N_, n_fields_)

  // record real time
  std::chrono::time_point<std::chrono::system_clock> real_t_beg_;
};

template <typename TRan>
void Solver::ini_rand(double noise_ini_cond, TRan &myran) {
  for (int i = 0; i < N_; i++) {
    f_[RHO + n_fields_ * i] = 1.;

    for(int k=1; k < n_fields_; k++) {
      f_[k + n_fields_ * i] = noise_ini_cond * (2. * myran.doub() - 1.);
    }
  }
}

class PseudoSpectralSolver: public Solver{
public:
  PseudoSpectralSolver(int Nx, int Ny, double Lx, double Ly, int n_fields);

  ~PseudoSpectralSolver();

  template <typename TRan>
  void ini_fields(double noise_ini_cond, TRan &myran, int do_antialiasing);

protected:

  // frequencies
  double *qx_ = nullptr;
  double *qy_ = nullptr;

  // fields
  int alloc_complex_;
  double *dxf_;
  double *dyf_;
  double *nonlinear_;
  fftw_complex *FFT_f_;
  fftw_complex *FFT_linear_;
  fftw_complex *FFT_nonlinear_;
  fftw_complex *FFT_support_deriv_;

  // plans of the FFTW
  fftw_plan forward_plan_;
  fftw_plan backward_plan_;

  double fftw_norm_;
};

template <typename TRan>
void PseudoSpectralSolver::ini_fields(double noise_ini_cond, TRan &myran,
                                      int do_antialiasing) {
  ini_rand(noise_ini_cond, myran);
  
  fftw_execute_dft_r2c(forward_plan_, f_, FFT_f_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_f_, do_antialiasing);
}