#include "BGL.h"
#include <cmath>
#include <iostream>

double cal_Pk(int k, double eta) {
  return exp(- k * k * eta * eta / 2.);
}

double cal_Ikq(int k, int q) {
  double Ikq;
  int tmp = k - 2 * q;
  if (abs(tmp) != 1) {
    Ikq = 4. / M_PI * (1. - tmp * sin(0.5 * M_PI * tmp)) / (1. - tmp * tmp);
  } else {
    Ikq = 2. / M_PI;
  }
  return Ikq;
}

double cal_Jkq(int k, int q, double eta) {
  return cal_Pk(k, eta) * cal_Ikq(k, q) - cal_Ikq(0, q);
}

BGL_Solver::BGL_Solver(int Nx, int Ny, double Lx, double Ly, int n_fields, double dt,
                       double eta, double rho0, double D0, int do_antialiasing):
                       PseudoSpectralSolver(Nx, Ny, Lx, Ly, n_fields),
                       do_antialiasing_(do_antialiasing) {
  double mu2 = cal_Pk(2, eta) - 1 + rho0 * (cal_Jkq(2, 0, eta) + cal_Jkq(2, 2, eta));
  mu1_ = cal_Pk(1, eta) - 1;
  mu1_rho_ = rho0 * (cal_Jkq(1, 0, eta) + cal_Jkq(1, 1, eta));
  xi_ = rho0 * rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) * cal_Jkq(2, 1, eta) / mu2;
  D_ = D0 - 1./(4 * mu2);
  kappa1_ = rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) / (2. * mu2);
  kappa2_ = rho0 * cal_Jkq(2, 1, eta) / (2. * mu2);
}

void BGL_Solver::eval_linear_part(double dt) const{
  const int Nyh = Ny_ / 2 + 1;
  for (int x = 0; x < Nx_; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int pos_rho = n_fields_ * (y + x_Nyh);
      const int pos_px = pos_rho + PX;
      const int pos_py = pos_rho + PY;
      const double q2 = qx_[x] * qx_[x] + qy_[y] * qy_[y];

      // rho
      FFT_linear_[pos_rho][RE] = FFT_f_[pos_rho][RE]
        + dt * (qx_[x] * FFT_f_[pos_px][IM] + qy_[y] * FFT_f_[pos_py][IM]);
      FFT_linear_[pos_rho][IM] = FFT_f_[pos_rho][IM]
        - dt * (qx_[x] * FFT_f_[pos_px][RE] + qy_[y] * FFT_f_[pos_py][RE]);
      
      const double mu1_minus_q2D = mu1_ - q2 * D_;
      // Px
      FFT_linear_[pos_px][RE] = FFT_f_[pos_px][RE] + dt * (
         0.5 * qx_[x] * FFT_f_[pos_rho][IM] + mu1_minus_q2D * FFT_f_[pos_px][RE]);
      FFT_linear_[pos_px][IM] = FFT_f_[pos_px][IM] + dt * (
        -0.5 * qx_[x] * FFT_f_[pos_rho][RE] + mu1_minus_q2D * FFT_f_[pos_px][IM]);
      
      // Py
      FFT_linear_[pos_py][RE] = FFT_f_[pos_py][RE] + dt * (
         0.5 * qy_[y] * FFT_f_[pos_rho][IM] + mu1_minus_q2D * FFT_f_[pos_py][RE]);
      FFT_linear_[pos_py][IM] = FFT_f_[pos_py][IM] + dt * (
        -0.5 * qy_[y] * FFT_f_[pos_rho][RE] + mu1_minus_q2D * FFT_f_[pos_py][IM]);
    }
  }
}

void BGL_Solver::eval_nonlinear_part(double dt) const{
  for (int i = 0; i < N_; i++) {
      const int pos = n_fields_ * i;
      const int pos_rho = pos + RHO;
      const int pos_px = pos + PX;
      const int pos_py = pos + PY;
      nonlinear_[pos_rho] = 0.;

      const double p_square = f_[pos_px] * f_[pos_px] + f_[pos_py] * f_[pos_py];
      const double xx_m_yy = dxf_[pos_px] - dyf_[pos_py];
      const double xx_p_yy = dxf_[pos_px] + dyf_[pos_py];
      const double xy_p_yx = dxf_[pos_py] + dyf_[pos_px];
      const double xy_m_yx = dxf_[pos_py] - dyf_[pos_px];
      const double tmp = mu1_rho_ * f_[pos_rho] - xi_ * p_square;
      nonlinear_[pos_px] = tmp * f_[pos_px]
        + kappa1_ * (f_[pos_px] * xx_m_yy + f_[pos_py] * xy_p_yx)
        + 2 * kappa2_ * (f_[pos_px] * xx_p_yy - f_[pos_py] * xy_m_yx);
      nonlinear_[pos_py] = tmp * f_[pos_py]
        + kappa1_ * (f_[pos_px] * xy_p_yx - f_[pos_py] * xx_m_yy)
        + 2 * kappa2_ * (f_[pos_px] * xy_m_yx + f_[pos_py] * xx_p_yy);
      nonlinear_[pos_px] *= dt;
      nonlinear_[pos_py] *= dt;  
    }
}

void BGL_Solver::eval_nonlinear_part(double dt, const double *RFx, const double *RFy) const {
    for (int i = 0; i < N_; i++) {
      const int pos = n_fields_ * i;
      const int pos_rho = pos + RHO;
      const int pos_px = pos + PX;
      const int pos_py = pos + PY;
      nonlinear_[pos_rho] = 0.;

      const double p_square = f_[pos_px] * f_[pos_px] + f_[pos_py] * f_[pos_py];
      const double xx_m_yy = dxf_[pos_px] - dyf_[pos_py];
      const double xx_p_yy = dxf_[pos_px] + dyf_[pos_py];
      const double xy_p_yx = dxf_[pos_py] + dyf_[pos_px];
      const double xy_m_yx = dxf_[pos_py] - dyf_[pos_px];
      const double tmp = mu1_rho_ * f_[pos_rho] - xi_ * p_square;
      nonlinear_[pos_px] = tmp * f_[pos_px]
        + kappa1_ * (f_[pos_px] * xx_m_yy + f_[pos_py] * xy_p_yx)
        + 2 * kappa2_ * (f_[pos_px] * xx_p_yy - f_[pos_py] * xy_m_yx)
        + RFx[i];
      nonlinear_[pos_py] = tmp * f_[pos_py]
        + kappa1_ * (f_[pos_px] * xy_p_yx - f_[pos_py] * xx_m_yy)
        + 2 * kappa2_ * (f_[pos_px] * xy_m_yx + f_[pos_py] * xx_p_yy)
        + RFy[i];
      nonlinear_[pos_px] *= dt;
      nonlinear_[pos_py] *= dt;  
    }
}

void BGL_Solver::one_step(double dt) const {
  eval_linear_part(dt);
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  // cal FFT of dxf
  eval_dx(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qx_);
  // cal dxf
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dxf_);

  // cal FFT of dyf
  eval_dy(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qy_);
  // cal dyf
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dyf_);

  eval_nonlinear_part(dt);
  fftw_execute_dft_r2c(forward_plan_, nonlinear_, FFT_nonlinear_);

  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_nonlinear_, do_antialiasing_);

  integrator_Euler(); // sum all
}

void BGL_Solver::one_step(double dt, const double *RFx, const double *RFy) const {
    eval_linear_part(dt);
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  // cal FFT of dxf
  eval_dx(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qx_);
  // cal dxf
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dxf_);

  // cal FFT of dyf
  eval_dy(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qy_);
  // cal dyf
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dyf_);

  eval_nonlinear_part(dt, RFx, RFy);

  fftw_execute_dft_r2c(forward_plan_, nonlinear_, FFT_nonlinear_);

  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_nonlinear_, do_antialiasing_);

  integrator_Euler(); // sum all
}

void BGL_Solver::save_phi(double t, const char *fname, int append) const {
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  save_order_para(t, fname, append);

  fftw_execute_dft_r2c(forward_plan_, f_, FFT_f_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_f_, do_antialiasing_);
}

void BGL_Solver::save_snap(double t, const char *fname, int append) const {
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  save_fields(t, fname, append);

  fftw_execute_dft_r2c(forward_plan_, f_, FFT_f_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_f_, do_antialiasing_);
}