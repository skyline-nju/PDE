/**
 * @file deriv.h
 * @author Yu Duan (duanyu100@yeah.net)
 * @brief 
 * @version 0.1
 * @date 2018-12-13
 * 
 * Calculate spatial derivatives by fft.
 *  
 * @copyright Copyright (c) 2018
 * 
 */

#pragma once
#include <fftw3.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
enum COMPONENT {RE, IM};
enum FIELD {RHO, PX, PY, NFILED};


void antialiasing_norm(int Nx, int Ny, int n_fields, double fftw_norm,
                       fftw_complex *fft_field, int anti_aliasing);

void eval_dx(int Nx, int Ny, int n_fields,
             const fftw_complex *fft_field,
             fftw_complex *fft_dfield, const double *qx);

void eval_dy(int Nx, int Ny, int n_fields,
             const fftw_complex *fft_field,
             fftw_complex *fft_dfield, const double *qy);

void eval_d2x(int Nx, int Ny, int n_fields,
              const fftw_complex *fft_field,
              fftw_complex *fft_dfield, const double *qx);

void eval_d2y(int Nx, int Ny, int n_fields,
              const fftw_complex *fft_field,
              fftw_complex *fft_dfield, const double *qy);