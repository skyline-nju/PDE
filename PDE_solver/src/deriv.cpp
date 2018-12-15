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