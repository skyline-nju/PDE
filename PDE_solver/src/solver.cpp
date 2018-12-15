#include "solver.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

Solver::Solver(int Nx, int Ny, double Lx, double Ly, int n_fields):
              Nx_(Nx), Ny_(Ny), N_(Nx * Ny), n_fields_(n_fields), Lx_(Lx), Ly_(Ly),
              alloc_real_(n_fields_ * N_) {
  // set parameters
  L_ = Lx_ * Ly_;
  dx_ = Lx_ / Nx_;
  dy_ = Ly_ / Ny_;
  f_ = fftw_alloc_real(alloc_real_);

  std::cout << "Nx = " << Nx_ << "\n";
  std::cout << "Ny - " << Ny_ << "\n";
  std::cout << "Lx = " << Lx_ << "\n";
  std::cout << "Ly = " << Ly_ << "\n";
  std::cout << "N = " << N_ << "\n";
  std::cout << std::endl;

  // record the starting time
  real_t_beg_ = std::chrono::system_clock::now();
}

Solver::~Solver() {
  // fftw_free(f_);
  std::cout << "Solver destructed" << std::endl;
}

void Solver::ini_from_file(const char *fname, int i_frame) {
  // to do ...
}

void Solver::save_order_para(double t, const char *fname, int append) const {
  double *order = new double[n_fields_];
  double *rho_arr = new double[N_];
  for (int k = 0; k < n_fields_; k++) {
    order[k] = 0.;
    for (int i = 0; i < N_; i++) {
      order[k] += f_[k + n_fields_ * i];
      if (k == 0) {
        rho_arr[i] = f_[n_fields_ * i];
      }
    }
    order[k] /= N_;
  }
  const double rho_min = *std::min_element(rho_arr, rho_arr + N_);

  std::ofstream fout;
  if (append) {
    fout.open(fname, std::ios::out|std::ios::app);
  } else {
    fout.open(fname, std::ios::out);
  }

  fout << t << "\t";
  for (int k = 0; k < n_fields_; k++) {
    fout << order[k] << "\t";
  }
  fout << rho_min << "\t";

  const auto t_now = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_time = t_now - real_t_beg_;
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
  delete[] order;
  delete[] rho_arr;
}

void Solver::save_fields(double t, const char *fname, int append) const {
  float *buf = new float[N_ * n_fields_ + 1];
  buf[0] = t;
  for (int i_field = 0; i_field < n_fields_; i_field++) {
    for (int x=0; x < Nx_; x++) {
      for (int y = 0; y < Ny_; y++) {
        int pos = y + x * Ny_ + i_field * N_;
        buf[1 + pos] = f_[i_field + n_fields_ * (y + x * Ny_)];
      }
    }
  }

  std::ofstream fout;
  if (append) {
    fout.open(fname, std::ios::binary|std::ios::app);
  } else {
    fout.open(fname, std::ios::binary);
  }
  fout.write(reinterpret_cast<char*>(buf), (N_ * n_fields_ + 1) * sizeof(float));
  fout.close();
  delete[] buf;
}

void set_frequencies(double *qx, double *qy, int Nx, int Ny, double Lx, double Ly) {
    double dkx = 2. * M_PI / Lx;
    double dky = 2. * M_PI / Ly;
    for(int x = 0; x < Nx; x++) {
      if(x <= Nx / 2) {
        qx[x] = dkx * x;
      } else {
        qx[x] = dkx * (x - Nx);
      }
    }
    for(int i=0;i <= Ny/2; i++) {
      qy[i] = dky * i;
    }
}

PseudoSpectralSolver::PseudoSpectralSolver(int Nx, int Ny, double Lx, double Ly, int n_fields):
                                           Solver(Nx, Ny, Lx, Ly, n_fields) {
  // set parameters
  fftw_norm_ = 1.0 / N_;

  // frequencies
  qx_ = new double[Nx];
  qy_ = new double[Ny / 2 + 1];
  set_frequencies(qx_, qy_, Nx_, Ny_, Lx_, Ly_);

  // fields
  alloc_complex_ = n_fields_ * Nx_ * (Ny_ / 2 + 1);
  std::cout << "max pos " << alloc_complex_ << std::endl; 
  dxf_ = fftw_alloc_real(alloc_real_);
  dyf_ = fftw_alloc_real(alloc_real_);
  nonlinear_ = fftw_alloc_real(alloc_real_);

  FFT_f_ = fftw_alloc_complex(alloc_complex_);
  FFT_linear_ = fftw_alloc_complex(alloc_complex_);
  FFT_nonlinear_ = fftw_alloc_complex(alloc_complex_);
  FFT_support_deriv_ = fftw_alloc_complex(alloc_complex_);
  std::cout << "a =" << FFT_support_deriv_[0][0] << std::endl;
  std::cout << "qx = " << qx_[0] << std::endl;

  // fftw plan
  int n[] = {Nx_, Ny_};
  forward_plan_ = fftw_plan_many_dft_r2c(2, n, n_fields_,
                                         f_, NULL, n_fields_, 1,
                                         FFT_f_, NULL, n_fields_, 1,
                                         FFTW_MEASURE);

  backward_plan_ = fftw_plan_many_dft_c2r(2, n, n_fields_,
                                          FFT_f_, NULL,n_fields_, 1,
                                          f_, NULL, n_fields_, 1,
                                          FFTW_MEASURE);
}

PseudoSpectralSolver::~PseudoSpectralSolver() {
  delete[] qx_;
  delete[] qy_;

  fftw_destroy_plan(forward_plan_);
  fftw_destroy_plan(backward_plan_);

  fftw_free(dxf_);
  fftw_free(dyf_);
  fftw_free(nonlinear_);
  fftw_free(FFT_f_);
  fftw_free(FFT_linear_);
  fftw_free(FFT_nonlinear_);
  fftw_free(FFT_support_deriv_);

  std::cout << "PseudoSpectralSolver destructed" << std::endl;
}

void PseudoSpectralSolver::integrator_Euler() const {
  for (int k = 0; k < alloc_complex_; k++) {
    FFT_f_[k][RE] = FFT_linear_[k][RE] + FFT_nonlinear_[k][RE];
    FFT_f_[k][IM] = FFT_linear_[k][IM] + FFT_nonlinear_[k][IM];
  }
}
