void Antialiasing_normalization(const int Nx, const int Ny, const double fftw_norm, \
                                fftw_complex *fft_field, const int anti_aliasing)
{
    int Nyh = Ny/2 + 1;
    
    //Can be adapted, usually 1/2 for cubic nonlinearities but 1/3 seems to work here
    int antialias_x = Nx/3;
    int antialias_y = Ny/3;
    
    for(int k=0;k<Nfields;k++)
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nyh;y++)
            {
                int pos = k + Nfields*(y + Nyh*x);
                
                if(anti_aliasing)
                {
                    if( (x <= antialias_x || x > Nx - antialias_x) && y <= antialias_y)
                    {
                        fft_field[pos][0] *= fftw_norm;
                        fft_field[pos][1] *= fftw_norm;
                    }
                    
                    else
                    {
                        fft_field[pos][0] = 0.;
                        fft_field[pos][1] = 0.;
                    }
                }
                
                else
                {
                    fft_field[pos][0] *= fftw_norm;
                    fft_field[pos][1] *= fftw_norm;
                }
            }
}

void Dx(const int Nx, const int Ny, const fftw_complex *fft_field, \
        fftw_complex *fft_dfield, const double *qx)
{
    int Nyh = Ny/2 + 1;
    
    for(int k=0;k<Nfields;k++)
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nyh;y++)
            {
                int pos = k + Nfields*(y + Nyh*x);
                
                fft_dfield[pos][0] = -qx[x]*fft_field[pos][1];
                fft_dfield[pos][1] = qx[x]*fft_field[pos][0];
            }
}

void Dy(const int Nx, const int Ny, const fftw_complex *fft_field, \
        fftw_complex *fft_dfield, const double *qy)
{
    int Nyh = Ny/2 + 1;
    
    for(int k=0;k<Nfields;k++)
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nyh;y++)
            {
                int pos = k + Nfields*(y + Nyh*x);
                
                fft_dfield[pos][0] = -qy[y]*fft_field[pos][1];
                fft_dfield[pos][1] = qy[y]*fft_field[pos][0];
            }
}

void D2x(const int Nx, const int Ny, const fftw_complex *fft_field, \
         fftw_complex *fft_dfield, const double *qx)
{
    int Nyh = Ny/2 + 1;
    
    for(int k=0;k<Nfields;k++)
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nyh;y++)
            {
                int pos = k + Nfields*(y + Nyh*x);

                fft_dfield[pos][0] = -qx[x]*qx[x]*fft_field[pos][0];
                fft_dfield[pos][1] = -qx[x]*qx[x]*fft_field[pos][1];
            }
}

void D2y(const int Nx, const int Ny, const fftw_complex *fft_field, \
         fftw_complex *fft_dfield, const double *qy)
{
    int Nyh = Ny/2 + 1;
    
    for(int k=0;k<Nfields;k++)
        for(int x=0;x<Nx;x++)
            for(int y=0;y<Nyh;y++)
            {
                int pos = k + Nfields*(y + Nyh*x);
                
                fft_dfield[pos][0] = -qy[y]*qy[y]*fft_field[pos][0];
                fft_dfield[pos][1] = -qy[y]*qy[y]*fft_field[pos][1];
            }
}

/********************************Dynamics*******************************/

void Linear_part(const fftw_complex *FFT_f, fftw_complex *FFT_Linear, \
                 const int Nx, const int Ny, const Equation_coefficients *C,\
                 const double *qx, const double *qy, const double dt, const bool append)
{
    int Nyh = Ny/2 + 1;
    
    for(int x=0;x<Nx;x++)
        for(int y=0;y<Nyh;y++)
        {
            int pos = Nfields*(y + Nyh*x);
            double q2 = (qx[x]*qx[x] + qy[y]*qy[y]);
            
            FFT_Linear[rho + pos][0] = (append>0?FFT_f[rho + pos][0]:0.) \
            + dt*( qx[x]*FFT_f[px + pos][1] + qy[y]*FFT_f[py + pos][1] );
            
            FFT_Linear[rho + pos][1] = (append>0?FFT_f[rho + pos][1]:0.)\
            - dt*( qx[x]*FFT_f[px + pos][0] + qy[y]*FFT_f[py + pos][0] );
            
#ifdef RANDOM_TORQUE
            FFT_Linear[px + pos][0] = (append>0?FFT_f[px + pos][0]:0.)\
            + dt*( qx[x]*FFT_f[rho + pos][1]/2. ) + C[0].mu1[Re]*FFT_f[px + pos][0];
            
            FFT_Linear[px + pos][1] = (append>0?FFT_f[px + pos][1]:0.)\
            - dt*( qx[x]*FFT_f[rho + pos][0]/2.) + C[0].mu1[Re]*FFT_f[px + pos][1];
            
            FFT_Linear[py + pos][0] = (append>0?FFT_f[py + pos][0]:0.)\
            + dt*( qy[y]*FFT_f[rho + pos][1]/2. ) + C[0].mu1[Re]*FFT_f[py + pos][0];
            
            FFT_Linear[py + pos][1] = (append>0?FFT_f[py + pos][1]:0.)\
            + dt*( -qy[y]*FFT_f[rho + pos][0]/2. ) + C[0].mu1[Re]*FFT_f[py + pos][1];
#else
            FFT_Linear[px + pos][0] = (append>0?FFT_f[px + pos][0]:0.)\
            + dt*( qx[x]*FFT_f[rho + pos][1]/2. ) - q2*C[0].D*FFT_f[px + pos][0];
            
            FFT_Linear[px + pos][1] = (append>0?FFT_f[px + pos][1]:0.)\
            - dt*( qx[x]*FFT_f[rho + pos][0]/2.) - q2*C[0].D*FFT_f[px + pos][1];
            
            FFT_Linear[py + pos][0] = (append>0?FFT_f[py + pos][0]:0.)\
            + dt*( qy[y]*FFT_f[rho + pos][1]/2. ) - q2*C[0].D*FFT_f[py + pos][0];
            
            FFT_Linear[py + pos][1] = (append>0?FFT_f[py + pos][1]:0.)\
            + dt*( -qy[y]*FFT_f[rho + pos][0]/2. ) - q2*C[0].D*FFT_f[py + pos][1];
#endif
        }
}

#ifdef RANDOM_TORQUE
void Nonlinear_part(const double *f, const double *dxf, const double *dyf, \
                    const double *d2xf, const double *d2yf, \
                    double *Nonlinear, const Equation_coefficients *C)
#else
void Nonlinear_part(const double *f, const double *dxf, const double *dyf, \
                    double *Nonlinear, const Equation_coefficients *C)
#endif
{
    for(int x=0;x<N;x++)
    {
        int pos = Nfields*x;
        
        double psq = f[px + pos]*f[px + pos] + f[py + pos]*f[py + pos];
        
        Nonlinear[rho + pos] = 0.;
 
#ifdef RANDOM_TORQUE
        Nonlinear[px + pos] = C[x].mu1_rho*f[rho + pos]*f[px + pos]\
        - C[x].mu1[Im]*f[py + pos]\
        - C[x].xi[Re]*psq*f[px + pos] + C[x].xi[Im]*psq*f[py + pos]\
        + C[x].D[Re]*(d2xf[px + pos] + d2yf[px + pos])\
        - C[x].D[Im]*(d2xf[py + pos] + d2yf[py + pos])\
        + C[x].kappa1[Re]*( f[px + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa1[Re]*( f[py + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        - C[x].kappa1[Im]*( f[px + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        + C[x].kappa1[Im]*( f[py + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa2[Re]*( f[px + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        - C[x].kappa2[Re]*( f[py + pos]*( dxf[py + pos] - dyf[px + pos] ) )\
        - C[x].kappa2[Im]*( f[px + pos]*( dxf[py + pos] - dyf[px + pos] ) )\
        - C[x].kappa2[Im]*( f[py + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        + C[x].alpha1[Re]*( dxf[px + pos] - dyf[py + pos] )\
        - C[x].alpha1[Im]*( dxf[py + pos] + dyf[px + pos] )\
        + C[x].beta1[Re]*(f[px + pos]*f[px + pos] - f[py + pos]*f[py + pos])\
        - 2.*C[x].beta1[Im]*f[px + pos]*f[py + pos];
    
        Nonlinear[py + pos] = C[x].mu1_rho*f[rho + pos]*f[py + pos]\
        + C[x].mu1[Im]*f[px + pos]\
        - C[x].xi[Re]*psq*f[py + pos] - C[x].xi[Im]*psq*f[px + pos]\
        + C[x].D[Re]*(d2xf[py + pos] + d2yf[py + pos])\
        + C[x].D[Im]*(d2xf[px + pos] + d2yf[px + pos])\
        + C[x].kappa1[Re]*( f[px + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        - C[x].kappa1[Re]*( f[py + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa1[Im]*( f[px + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa1[Im]*( f[py + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        + C[x].kappa2[Re]*( f[px + pos]*( dxf[py + pos] - dyf[px + pos] ) )\
        + C[x].kappa2[Re]*( f[py + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        + C[x].kappa2[Im]*( f[px + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        - C[x].kappa2[Im]*( f[py + pos]*( dxf[py + pos] - dyf[px + pos] ) )
        + C[x].alpha1[Re]*( dxf[py + pos] + dyf[px + pos] )\
        + C[x].alpha1[Im]*( dxf[px + pos] - dyf[py + pos] )\
        + 2.*C[x].beta1[Re]*f[px + pos]*f[py + pos]\
        + C[x].beta1[Im]*(f[px + pos]*f[px + pos] - f[py + pos]*f[py + pos]);
#else
        Nonlinear[px + pos] = C[x].mu1_rho*f[rho + pos]*f[px + pos]\
        + C[x].mu1[Re]*f[px + pos] - C[x].mu1[Im]*f[py + pos]\
        - C[x].xi*psq*f[px + pos]\
        + C[x].kappa1*( f[px + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa1*( f[py + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        + C[x].kappa2*( f[px + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        - C[x].kappa2*( f[py + pos]*( dxf[py + pos] - dyf[px + pos] ) )\
        + C[x].alpha1[Re]*( dxf[px + pos] - dyf[py + pos] )\
        - C[x].alpha1[Im]*( dxf[py + pos] + dyf[px + pos] )\
        + C[x].alpha2[Re]*( dxf[px + pos] + dyf[py + pos] )\
        - C[x].alpha2[Im]*( dxf[py + pos] - dyf[px + pos] )\
        + C[x].beta1[Re]*( f[px + pos]*f[px + pos] - f[py + pos]*f[py + pos] )\
        - 2.*C[x].beta1[Im]*f[px + pos]*f[py + pos]\
        + C[x].beta2[Re]*psq\
        + C[x].chi[Re]*f[rho + pos];
        
        Nonlinear[py + pos] = C[x].mu1_rho*f[rho + pos]*f[py + pos]\
        + C[x].mu1[Re]*f[py + pos] + C[x].mu1[Im]*f[px + pos]\
        - C[x].xi*psq*f[py + pos]\
        + C[x].kappa1*( f[px + pos]*( dxf[py + pos] + dyf[px + pos] ) )\
        - C[x].kappa1*( f[py + pos]*( dxf[px + pos] - dyf[py + pos] ) )\
        + C[x].kappa2*( f[px + pos]*( dxf[py + pos] - dyf[px + pos] ) )\
        + C[x].kappa2*( f[py + pos]*( dxf[px + pos] + dyf[py + pos] ) )\
        + C[x].alpha1[Re]*( dxf[py + pos] + dyf[px + pos] )\
        + C[x].alpha1[Im]*( dxf[px + pos] - dyf[py + pos] )\
        + C[x].alpha2[Re]*( dxf[py + pos] - dyf[px + pos] )\
        + C[x].alpha2[Im]*( dxf[px + pos] + dyf[py + pos] )\
        + 2.*C[x].beta1[Re]*f[px + pos]*f[py + pos]\
        + C[x].beta1[Im]*( f[px + pos]*f[px + pos] - f[py + pos]*f[py + pos] )\
        + C[x].beta2[Im]*psq\
        + C[x].chi[Im]*f[rho + pos];
#endif        
    }
}

//------------------------- Euler -------------------------//

void Euler_scheme(const int alloc_complex, fftw_complex *fnew, \
                  const fftw_complex *Linear, const fftw_complex *Nonlinear)
{
    for(int k=0;k<alloc_complex;k++)
    {
        double a = Linear[k][0] + Nonlinear[k][0];
        double b = Linear[k][1] + Nonlinear[k][1];
        
        fnew[k][0] = a;
        fnew[k][1] = b;
    }
}


//------------------------- RK4 -------------------------//

void Field_cpy(const int N, const fftw_complex *field, fftw_complex *cpy)
{
    for(int k=0;k<N;k++)
    {
        cpy[k][0] = field[k][0];
        cpy[k][1] = field[k][1];
    }
}

void RK4_scheme(const int alloc_complex, fftw_complex *f_new, \
                const fftw_complex *Linear, const fftw_complex *Nonlinear, \
                const fftw_complex *f, fftw_complex *beta, \
                const bool append, const double norm, \
                const bool compute_beta, const double norm_beta)
{
    if(append)
    {
        for(int k=0;k<alloc_complex;k++)
        {
            double a = Linear[k][0] + Nonlinear[k][0];
            double b = Linear[k][1] + Nonlinear[k][1];
            
            f_new[k][0] += norm*a;
            f_new[k][1] += norm*b;
            
            if(compute_beta)
            {
                beta[k][0] = f[k][0] + norm_beta*a;
                beta[k][1] = f[k][1] + norm_beta*b;
            }
        }
    }
    
    else
    {
        for(int k=0;k<alloc_complex;k++)
        {
            double a = Linear[k][0] + Nonlinear[k][0];
            double b = Linear[k][1] + Nonlinear[k][1];
            
            f_new[k][0] = f[k][0] + norm*a;
            f_new[k][1] = f[k][1] + norm*b;
            
            if(compute_beta)
            {
                beta[k][0] = f[k][0] + norm_beta*a;
                beta[k][1] = f[k][1] + norm_beta*b;
            }
        }
    }
}
