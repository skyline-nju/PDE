#include "BGL.h"
#include <cmath>
#include <iostream>


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

double cal_J(int k, int q, double eta2) {
  return cal_Pk(eta2, k) * cal_Ikq(k, q) - cal_Ikq(0, q);
}

void BGLSolverBase::one_step(double dt) const {
  //! the input array is overwrote during the c2r transform
  // cal dxf
  eval_dx(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qx_);
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dxf_);

  // cal dyf
  eval_dy(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qy_);
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dyf_);

  eval_linear_part(dt);
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  eval_nonlinear_part(dt);
  fftw_execute_dft_r2c(forward_plan_, nonlinear_, FFT_nonlinear_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_nonlinear_, do_antialiasing_);

  // sum all
  for (int k = 0; k < alloc_complex_; k++) {
    FFT_f_[k][RE] = FFT_linear_[k][RE] + FFT_nonlinear_[k][RE];
    FFT_f_[k][IM] = FFT_linear_[k][IM] + FFT_nonlinear_[k][IM];
  }
}

void BGLSolverBase::save_phi(double t, const char* fname, int append) const {
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  save_order_para(t, fname, append);

  fftw_execute_dft_r2c(forward_plan_, f_, FFT_f_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_f_, do_antialiasing_);
}

void BGLSolverBase::save_snap(double t, const char* fname, int append) const {
  fftw_execute_dft_c2r(backward_plan_, FFT_f_, f_);

  save_fields(t, fname, append);

  fftw_execute_dft_r2c(forward_plan_, f_, FFT_f_);
  antialiasing_norm(Nx_, Ny_, n_fields_, fftw_norm_, FFT_f_, do_antialiasing_);
}

void BGLSolverBase::eval_linear_part_pure(double dt, double mu1, double D) const {
  const int Nyh = Ny_ / 2 + 1;
  for (int x = 0; x < Nx_; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int pos_rho = n_fields_ * (y + x_Nyh);
      const int pos_px = pos_rho + PX;
      const int pos_py = pos_rho + PY;

      // rho
      FFT_linear_[pos_rho][RE] = FFT_f_[pos_rho][RE]
        + dt * (qx_[x] * FFT_f_[pos_px][IM] + qy_[y] * FFT_f_[pos_py][IM]);
      FFT_linear_[pos_rho][IM] = FFT_f_[pos_rho][IM]
        - dt * (qx_[x] * FFT_f_[pos_px][RE] + qy_[y] * FFT_f_[pos_py][RE]);
      
      const double q2 = qx_[x] * qx_[x] + qy_[y] * qy_[y];
      const double mu1_minus_q2D = mu1 - q2 * D;
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

BGL_pure::BGL_pure(int Nx, int Ny, double Lx, double Ly, int n_fields,
                   double eta, double rho0, double D0, int do_antialiasing):
                   BGLSolverBase(Nx, Ny, Lx, Ly, n_fields, do_antialiasing) {
  double mu2 = cal_Pk(2, eta) - 1 + rho0 * (cal_Jkq(2, 0, eta) + cal_Jkq(2, 2, eta));
  mu1_ = cal_Pk(1, eta) - 1;
  mu1_rho_ = rho0 * (cal_Jkq(1, 0, eta) + cal_Jkq(1, 1, eta));
  xi_ = rho0 * rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) * cal_Jkq(2, 1, eta) / mu2;
  D_ = D0 - 1./(4 * mu2);
  kappa1_ = rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) / (2. * mu2);
  kappa2_ = rho0 * cal_Jkq(2, 1, eta) / (2. * mu2);
}

void BGL_pure::eval_nonlinear_part(double dt) const {
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

BGL_RF::BGL_RF(int Nx, int Ny, double Lx, double Ly, int n_fields,
               double eta, double eps, double rho0, double D0, int do_antialiasing):
               BGLSolverBase(Nx, Ny, Lx, Ly, n_fields, do_antialiasing) {
  double mu2 = cal_Pk(2, eta) - 1 + rho0 * (cal_Jkq(2, 0, eta) + cal_Jkq(2, 2, eta));
  mu1_ = cal_Pk(1, eta) - 1;
  mu1_rho_ = rho0 * (cal_Jkq(1, 0, eta) + cal_Jkq(1, 1, eta));
  xi_ = rho0 * rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) * cal_Jkq(2, 1, eta) / mu2;
  D_ = D0 - 1./(4 * mu2);
  kappa1_ = rho0 * (cal_Jkq(1, 2, eta) + cal_Jkq(1, -1, eta)) / (2. * mu2);
  kappa2_ = rho0 * cal_Jkq(2, 1, eta) / (2. * mu2);

  // load random fields from file
  RFx_ = new double[N_];
  RFy_ = new double[N_];
  load_random_fields(eps, RFx_, RFy_, Nx, Ny, int(Lx), int(Ly));
}

void BGL_RF::eval_nonlinear_part(double dt) const {
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
      + RFx_[i];
    nonlinear_[pos_py] = tmp * f_[pos_py]
      + kappa1_ * (f_[pos_px] * xy_p_yx - f_[pos_py] * xx_m_yy)
      + 2 * kappa2_ * (f_[pos_px] * xy_m_yx + f_[pos_py] * xx_p_yy)
      + RFy_[i];
    nonlinear_[pos_px] *= dt;
    nonlinear_[pos_py] *= dt;  
  }
}

BGL_RP::BGL_RP(int Nx, int Ny, double Lx, double Ly, int n_fields,
              double eta, double eps, double rho0, double D0, int do_antialiasing):
              BGLSolverBase(Nx, Ny, Lx, Ly, n_fields, do_antialiasing) {

  P1_ = new double[N_];
  mu_rho_ = new double[N_];
  xi_ = new double[N_];
  nu_ = new double[N_];
  D_ = D0;

  kappa1_ = new double[N_];
  kappa2_2_ = new double[N_];

  alpha1_ = new double[N_];
  beta1_ = new double[N_];

  P2_x_ = fftw_alloc_real(N_);
  P2_y_ = fftw_alloc_real(N_);

  dx2f_ = fftw_alloc_real(alloc_real_);
  dy2f_ = fftw_alloc_real(alloc_real_);

  alpha3_ = new double[N_];
  beta3_ = new double[N_];

  double* rand_potentials = new double[N_];
  load_random_potential(eps, rand_potentials, Nx, Ny, int(Lx), int(Ly));

  double* P2 = new double[N_];

  double I20 = cal_Ikq(2, 0);
  double I22 = cal_Ikq(2, 2);
  double I21 = cal_Ikq(2, 1);
  // set coefficients
  for (int j = 0; j < N_; j++) {
    double noise2 = (eta + rand_potentials[j]) * (eta + rand_potentials[j]);
    P1_[j] = cal_P1(noise2);
    P2[j] = cal_P2(noise2);

    mu_rho_[j] = rho0 * (cal_J(1, 0, noise2) + cal_J(1, 1, noise2));
    
    double mu2 = P2[j] - 1 + rho0 * (cal_J(2, 0, noise2) + cal_J(2, 2, noise2));
    double J12_J1_1 = cal_J(1, 2, noise2) + cal_J(1, -1, noise2);
    double J21 = cal_J(2, 1, noise2);
    xi_[j] = rho0 * rho0 / mu2 * J12_J1_1 * J21;
    nu_[j] = -1. / (4. * mu2);
    kappa1_[j] = rho0 * J12_J1_1 / (2. * mu2);
    kappa2_2_[j] = rho0 * J21 / mu2;
    alpha1_[j] = (1. + rho0 * (I20 + I22)) / (4. * mu2 * mu2);
    beta1_[j] = rho0 * (I21 * mu2 - J21 * (1 + rho0 * (I20 + I22))) / (2 * mu2 * mu2);
    double J20_p_J22 = cal_J(2, 0, noise2) + cal_J(2, 2, noise2);
    alpha3_[j] = rho0 * J20_p_J22 / (4 * mu2 * mu2);
    beta3_[j] = -rho0 * rho0 * J21 * J20_p_J22 / (2 * mu2 * mu2);
  }

  // cal \partial_x P_2 and \partial_y P_2
  eval_df(Nx, Ny, P2, qx_, qy_, P2_x_, P2_y_);

  delete[] rand_potentials;
  delete[] P2;
}

BGL_RP::~BGL_RP() {
  delete[] P1_;
  delete[] mu_rho_;
  delete[] xi_;
  delete[] nu_;

  delete[] kappa1_;
  delete[] kappa2_2_;
  delete[] alpha1_;
  delete[] beta1_;
  delete[] alpha3_;
  delete[] beta3_;

  fftw_free(P2_x_);
  fftw_free(P2_y_);
  fftw_free(dx2f_);
  fftw_free(dy2f_);
}

void BGL_RP::eval_linear_part(double dt) const {
  const int Nyh = Ny_ / 2 + 1;
  for (int x = 0; x < Nx_; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int pos_rho = n_fields_ * (y + x_Nyh);
      const int pos_px = pos_rho + PX;
      const int pos_py = pos_rho + PY;

      // rho
      FFT_linear_[pos_rho][RE] = FFT_f_[pos_rho][RE]
        + dt * (qx_[x] * FFT_f_[pos_px][IM] + qy_[y] * FFT_f_[pos_py][IM]);
      FFT_linear_[pos_rho][IM] = FFT_f_[pos_rho][IM]
        - dt * (qx_[x] * FFT_f_[pos_px][RE] + qy_[y] * FFT_f_[pos_py][RE]);

      const double q2 = qx_[x] * qx_[x] + qy_[y] * qy_[y];
      const double coeff = -1. - q2 * D_;
      // Px
      FFT_linear_[pos_px][RE] = FFT_f_[pos_px][RE] + dt * (
        0.5 * qx_[x] * FFT_f_[pos_rho][IM] + coeff * FFT_f_[pos_px][RE]);
      FFT_linear_[pos_px][IM] = FFT_f_[pos_px][IM] + dt * (
        -0.5 * qx_[x] * FFT_f_[pos_rho][RE] + coeff * FFT_f_[pos_px][IM]);

      // Py
      FFT_linear_[pos_py][RE] = FFT_f_[pos_py][RE] + dt * (
        0.5 * qy_[y] * FFT_f_[pos_rho][IM] + coeff * FFT_f_[pos_py][RE]);
      FFT_linear_[pos_py][IM] = FFT_f_[pos_py][IM] + dt * (
        -0.5 * qy_[y] * FFT_f_[pos_rho][RE] + coeff * FFT_f_[pos_py][IM]);
    }
  }

  // cal \Delta f
  eval_d2x(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qx_);
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dx2f_);
  eval_d2y(Nx_, Ny_, n_fields_, FFT_f_, FFT_support_deriv_, qy_);
  fftw_execute_dft_c2r(backward_plan_, FFT_support_deriv_, dy2f_);
}

void BGL_RP::eval_nonlinear_part(double dt) const {
  for (int i = 0; i < N_; i++) {
    const int pos = n_fields_ * i;
    const int pos_rho = pos + RHO;
    const int pos_px = pos + PX;
    const int pos_py = pos + PY;
    nonlinear_[pos_rho] = 0.;

    const double px = f_[pos_px];
    const double py = f_[pos_py];

    const double px_x = dxf_[pos_px];
    const double py_x = dxf_[pos_py];
    const double px_y = dyf_[pos_px];
    const double py_y = dyf_[pos_py];

    const double px_px_x = px * px_x;
    const double px_px_y = px * px_y;
    const double px_py_x = px * py_x;
    const double px_py_y = px * py_y;

    const double py_px_x = py * px_x;
    const double py_px_y = py * px_y;
    const double py_py_x = py * py_x;
    const double py_py_y = py * py_y;

    const double px2 = px * px;
    const double py2 = py * py;
    const double coeff_f1 = P1_[i] + mu_rho_[i] * f_[pos_rho] - xi_[i] * (px2 + py2);

    const double k1_p_2k2 = kappa1_[i] + kappa2_2_[i];
    const double k1_m_2k2 = kappa1_[i] - kappa2_2_[i];
    const double A = alpha1_[i] * (px_x - py_y) + beta1_[i] * (px2 - py2);
    const double B = alpha1_[i] * (py_x + px_y) + 2. * beta1_[i] * px * py;
    nonlinear_[pos_px] = coeff_f1 * px
      + nu_[i] * (dx2f_[pos_px] + dy2f_[pos_px])
      + k1_p_2k2 * (px_px_x + py_px_y) + k1_m_2k2 * (py_py_x - px_py_y);
      + A * P2_x_[i] + B * P2_y_[i];

    nonlinear_[pos_py] = coeff_f1 * py
      + nu_[i] * (dx2f_[pos_py] + dy2f_[pos_py])
      + k1_p_2k2 * (px_py_x + py_py_y) + k1_m_2k2 * (px_px_y - py_px_x);
      - A * P2_y_[i] + B * P2_x_[i];

    //const double C = alpha3_[i] * (px_x - py_y) + beta3_[i] * (px2 - py2);
    //const double D = alpha3_[i] * (py_x + px_y) + 2. * beta3_[i] * px * py;
    //const double rho_x = dxf_[pos_rho];
    //const double rho_y = dyf_[pos_rho];
    //nonlinear_[pos_px] += C * rho_x + D * rho_y;
    //nonlinear_[pos_py] += (-C * rho_y + D * rho_x);

    nonlinear_[pos_px] *= dt;
    nonlinear_[pos_py] *= dt;
  }
}
