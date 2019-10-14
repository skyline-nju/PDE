/**
 * @file BGL.h
 * @author Yu Duan (duanyu100@yeah.net)
 * @brief 
 * @version 0.1
 * @date 2018-12-14
 * 
 * Solver for the BGL hydrodynamic equations
 * 
 * @copyright Copyright (c) 2018
 * 
 */

#pragma once
#include "solver.h"

/**
 * @brief Solver for the BGL hydrodynamic equations using the pseudo spectral
 * method.
 * 
 */
class BGL_Solver: public PseudoSpectralSolver{
public:
  BGL_Solver(int Nx, int Ny, double Lx, double Ly, int n_fields,
             double dt, double eta, double rho0, double D0,
             int do_antialiasing);
  
  BGL_Solver(int Nx, int Ny, double Lx, double Ly, int n_fields,
             double dt, double eta, double eta_sd, double rho0, double D0,
             int do_antialiasing);

  void eval_linear_part(double dt) const;

  void eval_nonlinear_part(double dt) const;

  void eval_nonlinear_part(double dt, const double *RFx, const double *RFy) const;

  void one_step(double dt) const;

  void one_step(double dt, const double *RFx, const double *RFy) const;

  void save_phi(double t, const char *fname, int append) const;

  void save_snap(double t, const char *fname, int append) const;

protected:
  double mu1_;
  double mu1_rho_;
  double xi_;
  double D_;
  double kappa1_;
  double kappa2_;

  int n_mod_ = 3;

  int do_antialiasing_;
};

