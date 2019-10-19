#include "deriv.h"
//#include <fftw3.h>
#include <iostream>

void antialiasing_norm(int Nx, int Ny, int n_fields, double fftw_norm,
                       fftw_complex *fft_field, int anti_aliasing) {
  const int Nyh = Ny / 2 + 1;
  //Can be adapted, usually 1/2 for cubic nonlinearities but 1/3 seems to work here
  const int antialias_x = Nx/3;
  const int antialias_y = Ny/3;

  for (int x = 0; x < Nx; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int tmp = n_fields * (y + x_Nyh);
      for (int k = 0; k < n_fields; k++) {
        int pos = k + tmp;
        if (anti_aliasing) {
          if ((x <= antialias_x || x > Nx - antialias_x) && y <= antialias_y) {
            fft_field[pos][RE] *= fftw_norm;
            fft_field[pos][IM] *= fftw_norm;
          } else {
            fft_field[pos][RE] = 0.;
            fft_field[pos][IM] = 0.;
          }
        } else {
          fft_field[pos][RE] *= fftw_norm;
          fft_field[pos][IM] *= fftw_norm;
        }
      }
    }
  }
}

void eval_dx(int Nx, int Ny, int n_fields, const fftw_complex *fft_field,
             fftw_complex *fft_dfield, const double *qx) {
  int Nyh = Ny / 2 + 1;
  for (int x=0; x < Nx; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int tmp = n_fields * (y + x_Nyh);
      for (int k = 0; k < n_fields; k++) {
        int pos = k + tmp;
        fft_dfield[pos][RE] = -qx[x] * fft_field[pos][IM];
        fft_dfield[pos][IM] =  qx[x] * fft_field[pos][RE];
      }
    }
  }

}

void eval_dy(int Nx, int Ny, int n_fields, const fftw_complex *fft_field,
             fftw_complex *fft_dfield, const double *qy) {
  int Nyh = Ny / 2 + 1;
  for (int x=0; x < Nx; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int tmp = n_fields * (y + x_Nyh);
      for (int k = 0; k < n_fields; k++) {
        int pos = k + tmp;
        fft_dfield[pos][RE] = -qy[y] * fft_field[pos][IM];
        fft_dfield[pos][IM] =  qy[y] * fft_field[pos][RE];
      }
    }
  }
}

void eval_d2x(int Nx, int Ny, int n_fields, const fftw_complex *fft_field,
              fftw_complex *fft_dfield, const double *qx) {
  int Nyh = Ny / 2 + 1;

  for (int x=0; x < Nx; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int tmp = n_fields * (y + x_Nyh);
      for (int k = 0; k < n_fields; k++) {
        int pos = k + tmp;
        fft_dfield[pos][RE] = -qx[x] * qx[x] * fft_field[pos][RE];
        fft_dfield[pos][IM] = -qx[x] * qx[x] * fft_field[pos][IM];
      }
    }
  }
}

void eval_d2y(int Nx, int Ny, int n_fields, const fftw_complex *fft_field,
              fftw_complex *fft_dfield, const double *qy) {
  int Nyh = Ny / 2 + 1;

  for (int x=0; x < Nx; x++) {
    const int x_Nyh = x * Nyh;
    for (int y = 0; y < Nyh; y++) {
      const int tmp = n_fields * (y + x_Nyh);
      for (int k = 0; k < n_fields; k++) {
        int pos = k + tmp;
        fft_dfield[pos][RE] = -qy[y] * qy[y] * fft_field[pos][RE];
        fft_dfield[pos][IM] = -qy[y] * qy[y] * fft_field[pos][IM];
      }
    }
  }
}

void eval_df(int Nx, int Ny, double* f, const double* qx, const double* qy,
             double* fx, double* fy) {
  int alloc_real = Nx * Ny;
  int alloc_complex = Nx * (Ny / 2 + 1);
  double norm = 1. / (Nx * Ny);
  double* f_tmp = fftw_alloc_real(alloc_real);
  fftw_complex* FFT_f = fftw_alloc_complex(alloc_complex);
  fftw_complex* FFT_fx = fftw_alloc_complex(alloc_complex);  // \partial_x FFT_f
  fftw_complex* FFT_fy = fftw_alloc_complex(alloc_complex);  // \partial_y FFT_f

  fftw_plan forward = fftw_plan_dft_r2c_2d(Nx, Ny, f_tmp, FFT_f, FFTW_ESTIMATE);
  fftw_plan backward = fftw_plan_dft_c2r_2d(Nx, Ny, FFT_fx, fx, FFTW_MEASURE);

  fftw_execute_dft_r2c(forward, f, FFT_f);
  
  eval_dx(Nx, Ny, 1, FFT_f, FFT_fx, qx);
  eval_dy(Nx, Ny, 1, FFT_f, FFT_fy, qy);

  fftw_execute_dft_c2r(backward, FFT_fx, fx);
  fftw_execute_dft_c2r(backward, FFT_fy, fy);

  for (int x = 0; x < Nx; x++) {
    const int x_Ny = x * Ny;
    for (int y = 0; y < Ny; y++) {
      const int pos = y + x_Ny;
      fx[pos] *= norm;
      fy[pos] *= norm;
    }
  }

  fftw_destroy_plan(forward);
  fftw_destroy_plan(backward);
  fftw_free(f_tmp);
  fftw_free(FFT_f);
  fftw_free(FFT_fx);
  fftw_free(FFT_fy);
}
