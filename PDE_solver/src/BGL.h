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
#include "disorder.h"

class BGLSolverBase: public PseudoSpectralSolver {
public:
  BGLSolverBase(int Nx, int Ny, double Lx, double Ly, int n_fields, int do_antialiasing):
                PseudoSpectralSolver(Nx, Ny, Lx, Ly, n_fields),
                do_antialiasing_(do_antialiasing) {}
  
  virtual void eval_linear_part(double dt) const=0;

  virtual void eval_nonlinear_part(double dt) const=0;

  void one_step(double dt) const;

  void save_phi(double t, const char* fname, int append) const;

  void save_snap(double t, const char* fname, int append) const;

  void eval_linear_part_pure(double dt, double mu1, double D) const;

protected:
  int n_mod_ = 3;
  int do_antialiasing_;
};

class BGL_pure: public BGLSolverBase {
public:
  BGL_pure(int Nx, int Ny, double Lx, double Ly, int n_fields,
           double eta, double rho0, double D0, int do_antialiasing);

  void eval_linear_part(double dt) const {
    eval_linear_part_pure(dt, mu1_, D_);
  }

  void eval_nonlinear_part(double dt) const;

private:
  double mu1_;
  double mu1_rho_;
  double xi_;
  double D_;
  double kappa1_;
  double kappa2_;
};

class BGL_RF: public BGLSolverBase {
public:
  BGL_RF(int Nx, int Ny, double Lx, double Ly, int n_fields,
         double eta, double eps, double rho0, double D0, int do_antialiasing);

  ~BGL_RF() {
    delete[] RFx_;
    delete[] RFy_;
  }

  void eval_linear_part(double dt) const {
    eval_linear_part_pure(dt, mu1_, D_);
  }

  void eval_nonlinear_part(double dt) const;

private:
  double mu1_;
  double mu1_rho_;
  double xi_;
  double D_;
  double kappa1_;
  double kappa2_;

  double* RFx_;
  double* RFy_;
};

