/****************************Coefficients***********************/
#ifndef M_PI
 #define M_PI 3.14159265358979323846
 #endif

double Int(const int k, const int q)
{
    double mode = -q + 0.5*k;
    double integral;
    if(mode != 0.0 && fabs(mode) != 0.5 )
        integral = 4.0*(1.0 - 2.0*mode*sin(M_PI*mode)) / (1.0-4.0*mode*mode);
    else if( fabs(mode) == 0.5 )
        integral = 2.0;
    else
        integral = 4.0;
    return integral/M_PI;
}

void normalize_torque(int Nx, int Ny, double *gamma) {
    double ave = 0;
    int tot = Nx * Ny;
    for (int i = 0; i < tot; i++) {
        ave += gamma[i];
    }
    ave /= tot;
    std::cout << "mean torque before normalizing: " << ave << std::endl;
    double mean = 0.;
    for (int i = 0; i < tot; i++) {
        gamma[i] -= ave;
        mean += gamma[i];
    }
    mean /= tot;
    std::cout << "mean torque after normalizing: " << mean << std::endl;
}

void normalize_field(int Nx, int Ny, double *expgamma, double *gamma) {
    double ave[2] = {0, 0};
    int tot = Nx * Ny;
    for (int i = 0; i < tot; i++) {
        expgamma[Re + i * 2] = cos(gamma[i]);
        expgamma[Im + i * 2] = sin(gamma[i]);
        ave[Re] += expgamma[Re + i * 2];
        ave[Im] += expgamma[Im + i * 2];
    }
    ave[Re] /= tot;
    ave[Im] /= tot;

    for (int i = 0; i < tot; i++) {
        expgamma[Re + i * 2] -= ave[Re];
        expgamma[Im + i * 2] -= ave[Im];
        gamma[i] = atan2(expgamma[Im + i * 2], expgamma[Re + i * 2]);
    }
}

void Coefficients_computation(Equation_coefficients *C, const int Nx, const int Ny,\
                              const double density, const double eta, \
                              const double D0, const double zeta, \
                              const double dt, \
                              const int load_quenched, const char *name_quenched)
{
    double Intpos[Nmod+1][Nmod+1], Intneg[Nmod+1][Nmod+1], p[Nmod+1];
    
    for(int k=0;k<=Nmod;k++)
    {
        p[k] = exp(-k*k*eta*eta/2.);
        for(int q=0;q<=Nmod;q++)
        {
            Intpos[k][q] = Int(k,q);
            Intneg[k][q] = Int(k,-q);
        }
    }
    
    double mu2 = p[2] - 1. + ( p[2]*(Intpos[2][0] + Intpos[2][2]) \
                              - Intpos[0][2] - Intpos[0][0] )*density;
    double x12 = density*(p[1]*(Intpos[1][2] + Intneg[1][1]) \
                          - Intpos[0][2] - Intneg[0][1]);
    double j21 = density*(p[2]*Intpos[2][1] - Intpos[0][1]);
    
    int x,y;
    
    //-------------- Compute/load quench disorder-------------------//
    double norm = 1./(double)(Nx*Ny);
    
    double *gamma = new double[Nx*Ny];
#ifndef RANDOM_TORQUE
    double *expgamma = new double[2*Nx*Ny];
#endif
    
    int Nyh = Ny/2 + 1;
    double *dxgamma = new double[Nx*Ny];
    double *dygamma = new double[Nx*Ny];
    
    fftw_complex *FFT_gamma = fftw_alloc_complex(Nx*Nyh);
    fftw_complex *support = fftw_alloc_complex(Nx*Nyh);
    
    fftw_plan forward_gamma = fftw_plan_dft_r2c_2d(Nx,Ny,gamma,FFT_gamma,FFTW_MEASURE);
    fftw_plan backward_gamma = fftw_plan_dft_c2r_2d(Nx,Ny,FFT_gamma,gamma,FFTW_MEASURE);
    fftw_plan dx_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dxgamma,FFTW_MEASURE);
    fftw_plan dy_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dygamma,FFTW_MEASURE);
    
    if(!load_quenched)
    {
#ifdef RANDOM_TORQUE
        double avg_gamma = 0.;
#else
        double avg_field[2] = {0.,0.};
#endif
        for(x=0;x<Nx;x++)
            for(y=0;y<Ny;y++)
            {
                int pos = y + Ny*x;
                gamma[pos] = M_PI*(2.*PRNG_double01() - 1.);
     
#ifdef RANDOM_TORQUE
                avg_gamma += gamma[pos];
#else
                expgamma[Re + 2*pos] = cos(gamma[pos]);
                expgamma[Im + 2*pos] = sin(gamma[pos]);
                
                avg_field[Re] += expgamma[Re + 2*pos];
                avg_field[Im] += expgamma[Im + 2*pos];
#endif
            }
#ifdef RANDOM_TORQUE
        avg_gamma *= norm;
        for(x=0;x<Nx*Ny;x++)
            gamma[x] -= avg_gamma;
        
        ofstream out(name_quenched,ios::out|ios::binary);
        out.write((char*)gamma,Nx*Ny*sizeof(double));
        out.close();
#else
        avg_field[Re] *= norm;
        avg_field[Im] *= norm;
        for(x=0;x<Nx*Ny;x++)
        {
            expgamma[Re + 2*x] -= avg_field[Re];
            expgamma[Im + 2*x] -= avg_field[Im];
            gamma[x] = atan2(expgamma[Im + 2*x],expgamma[Re + 2*x]);
        }
        
        ofstream out(name_quenched,ios::out|ios::binary);
        out.write((char*)expgamma,2*Nx*Ny*sizeof(double));
        out.close();
#endif
    }
    else
    {
        ifstream in(name_quenched,ios::in|ios::binary);
        if(in.fail())
            cout << "Couldn't load quenched, no file found." << endl;
        else {
#ifdef RANDOM_TORQUE
            in.read((char*)gamma,Nx*Ny*sizeof(double));
#else
            in.read((char*)expgamma,2*Nx*Ny*sizeof(double));
            for(x=0;x<Nx*Ny;x++)
                gamma[x] = atan2(expgamma[Im + 2*x],expgamma[Re + 2*x]);
#endif
            in.close();
        }
    }
    
    //-------------- Compute gradient of gamma-------------------//
    
    fftw_execute(forward_gamma);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qx[x]*FFT_gamma[pos][1];
            support[pos][1] = qx[x]*FFT_gamma[pos][0];
        }
    fftw_execute(dx_plan);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qy[y]*FFT_gamma[pos][1];
            support[pos][1] = qy[y]*FFT_gamma[pos][0];
        }
    fftw_execute(dy_plan);
    
    fftw_execute(backward_gamma);
    
    fftw_destroy_plan(forward_gamma);
    fftw_destroy_plan(backward_gamma);
    fftw_destroy_plan(dx_plan);
    fftw_destroy_plan(dy_plan);
    
    fftw_free(FFT_gamma);
    fftw_free(support);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            gamma[pos] *= norm;
            dxgamma[pos] *= norm;
            dygamma[pos] *= norm;
        }
    
    //---------- Compute coefficients according to type of quenched --------------//
    
#ifdef RANDOM_TORQUE
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            C[pos].mu1[Re] = dt*(p[1] - 1.);
            C[pos].mu1[Im] = dt*zeta*gamma[pos];
            
            double den_sq = 1./(mu2*mu2 + 4.*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].xi[Re] = dt*x12*j21*mu2*den_sq;
            C[pos].xi[Im] = -dt*x12*j21*2.*zeta*gamma[pos]*den_sq;
            
            C[pos].D[Re] = dt*D0 - dt*mu2*den_sq/4.;
            C[pos].D[Im] = dt*zeta*gamma[pos]*den_sq/2.;
            
            C[pos].kappa1[Re] = dt*x12*mu2*den_sq/2.;
            C[pos].kappa1[Im] = -dt*x12*zeta*gamma[pos]*den_sq;
            
            C[pos].kappa2[Re] = dt*j21*mu2*den_sq;
            C[pos].kappa2[Im] = -dt*2.*j21*zeta*gamma[pos]*den_sq;
            
            den_sq = 1./( (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*\
                         (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos]) + \
                         16.*mu2*mu2*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].alpha1[Re] = dt*zeta*(4.*mu2*zeta*gamma[pos]*dxgamma[pos] +
                (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dygamma[pos])*den_sq/2.;
            C[pos].alpha1[Im] = dt*zeta*(-4.*mu2*zeta*gamma[pos]*dygamma[pos] +
                    (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dxgamma[pos])*den_sq/2.;
            
            C[pos].beta1[Re] = -2.*j21*C[pos].alpha1[Re];
            C[pos].beta1[Im] = -2.*j21*C[pos].alpha1[Im];
        }
#else
    double mu[2] = {0.,0.};
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            double cosg = expgamma[Re + 2*pos], sing = expgamma[Im + 2*pos];
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            
            double mur = - zeta*dt*( sing*dxgamma[pos] - cosg*dygamma[pos] )/(2.*mu2);
            double mui = dt*zeta*( cosg*dxgamma[pos] + sing*dygamma[pos] )/(2.*mu2);
            
            C[pos].mu1[Re] = dt*(p[1] - 1. + zeta*zeta/(2.*mu2)) + mur;
            C[pos].mu1[Im] = mui;
            
            mu[Re] += mur; //Need to compensate this part of the linear coefficient
            mu[Im] += mui; // to remove chirality
            
            C[pos].xi = dt*x12*j21/mu2;
            C[pos].D = dt*D0 - dt/(4.*mu2);
            C[pos].kappa1 = dt*x12/(2.*mu2);
            C[pos].kappa2 = dt*j21/mu2;
            
            C[pos].alpha1[Re] = -dt*zeta*cosg/(4.*mu2);
            C[pos].alpha1[Im] = dt*zeta*sing/(4.*mu2);
            
            C[pos].alpha2[Re] = dt*zeta*cosg/(2.*mu2);
            C[pos].alpha2[Im] = dt*zeta*sing/(2.*mu2);
            
            C[pos].beta1[Re] = dt*zeta*j21*cosg/(2.*mu2);
            C[pos].beta1[Im] = -dt*zeta*j21*sing/(2.*mu2);
            
            C[pos].beta2[Re] = -dt*zeta*x12*cosg/mu2;
            C[pos].beta2[Im] = -dt*zeta*x12*sing/mu2;
            
            C[pos].chi[Re] = dt*zeta*cosg/2.;
            C[pos].chi[Im] = dt*zeta*sing/2.;
        }
    
    mu[Re] *= norm;
    mu[Im] *= norm;
    
    for(x=0;x<Nx*Ny;x++)
    {
        C[x].mu1[Re] -= mu[Re];
        C[x].mu1[Im] -= mu[Im];
    }
#endif
    
    //------------ Save quenched --------------//
    
#ifndef RANDOM_TORQUE
    delete[] expgamma;
#endif
    
    delete[] gamma;
    delete[] dxgamma;
    delete[] dygamma;
}

void Coefficients_computation(Equation_coefficients *C, int Nx, int Ny, double Lx, double Ly,
                              double density, double eta, double D0, double zeta,
                              double dt, int load_quenched, const char *name_quenched)
{
    double Intpos[Nmod+1][Nmod+1], Intneg[Nmod+1][Nmod+1], p[Nmod+1];
    
    for(int k=0;k<=Nmod;k++)
    {
        p[k] = exp(-k*k*eta*eta/2.);
        for(int q=0;q<=Nmod;q++)
        {
            Intpos[k][q] = Int(k,q);
            Intneg[k][q] = Int(k,-q);
        }
    }
    
    double mu2 = p[2] - 1. + ( p[2]*(Intpos[2][0] + Intpos[2][2]) \
                              - Intpos[0][2] - Intpos[0][0] )*density;
    double x12 = density*(p[1]*(Intpos[1][2] + Intneg[1][1]) \
                          - Intpos[0][2] - Intneg[0][1]);
    double j21 = density*(p[2]*Intpos[2][1] - Intpos[0][1]);
    
    int x,y;
    
    //-------------- Compute/load quench disorder-------------------//
    double norm = 1./(double)(Nx*Ny);
    
    double *gamma = new double[Nx*Ny];
#ifndef RANDOM_TORQUE
    double *expgamma = new double[2*Nx*Ny];
#endif
    
    int Nyh = Ny/2 + 1;
    double *dxgamma = new double[Nx*Ny];
    double *dygamma = new double[Nx*Ny];
    
    fftw_complex *FFT_gamma = fftw_alloc_complex(Nx*Nyh);
    fftw_complex *support = fftw_alloc_complex(Nx*Nyh);
    
    fftw_plan forward_gamma = fftw_plan_dft_r2c_2d(Nx,Ny,gamma,FFT_gamma,FFTW_MEASURE);
    fftw_plan backward_gamma = fftw_plan_dft_c2r_2d(Nx,Ny,FFT_gamma,gamma,FFTW_MEASURE);
    fftw_plan dx_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dxgamma,FFTW_MEASURE);
    fftw_plan dy_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dygamma,FFTW_MEASURE);
    
    if(!load_quenched) {
        int Mx = int(Lx);
        int My = int(Ly);
        int block_size_x = Nx / Mx;
        int block_size_y = Ny / My;
        for (x= 0; x < Mx; x++) {
            for (y = 0; y < My; y++) {
                double tmp = M_PI * (2. * PRNG_double01() - 1.);
                for (int i = 0; i < block_size_x; i++) {
                    for (int j = 0; j < block_size_y; j++) {
                        int pos = (y * block_size_y + j) + Ny * (x * block_size_x + i);
                        gamma[pos] = tmp;
                    }
                }
            }
        }
    } else {
        ifstream in(name_quenched,ios::in|ios::binary);
        if(in.fail()) {
            cout << "Couldn't load quenched, no file found." << endl;
        } else {
            in.read((char*)gamma,Nx*Ny*sizeof(double));
            in.close();
        }
    }
#ifdef RANDOM_TORQUE
    normalize_torque(Nx, Ny, gamma);
#else
    normalize_field(Nx, Ny, expgamma, gamma);
#endif
    
    //-------------- Compute gradient of gamma-------------------//
    
    fftw_execute(forward_gamma);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qx[x]*FFT_gamma[pos][1];
            support[pos][1] = qx[x]*FFT_gamma[pos][0];
        }
    fftw_execute(dx_plan);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qy[y]*FFT_gamma[pos][1];
            support[pos][1] = qy[y]*FFT_gamma[pos][0];
        }
    fftw_execute(dy_plan);
    
    fftw_execute(backward_gamma);
    
    fftw_destroy_plan(forward_gamma);
    fftw_destroy_plan(backward_gamma);
    fftw_destroy_plan(dx_plan);
    fftw_destroy_plan(dy_plan);
    
    fftw_free(FFT_gamma);
    fftw_free(support);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            gamma[pos] *= norm;
            dxgamma[pos] *= norm;
            dygamma[pos] *= norm;
        }
    
    //---------- Compute coefficients according to type of quenched --------------//
    
#ifdef RANDOM_TORQUE
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            C[pos].mu1[Re] = dt*(p[1] - 1.);
            C[pos].mu1[Im] = dt*zeta*gamma[pos];
            
            double den_sq = 1./(mu2*mu2 + 4.*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].xi[Re] = dt*x12*j21*mu2*den_sq;
            C[pos].xi[Im] = -dt*x12*j21*2.*zeta*gamma[pos]*den_sq;
            
            C[pos].D[Re] = dt*D0 - dt*mu2*den_sq/4.;
            C[pos].D[Im] = dt*zeta*gamma[pos]*den_sq/2.;
            
            C[pos].kappa1[Re] = dt*x12*mu2*den_sq/2.;
            C[pos].kappa1[Im] = -dt*x12*zeta*gamma[pos]*den_sq;
            
            C[pos].kappa2[Re] = dt*j21*mu2*den_sq;
            C[pos].kappa2[Im] = -dt*2.*j21*zeta*gamma[pos]*den_sq;
            
            den_sq = 1./( (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*\
                         (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos]) + \
                         16.*mu2*mu2*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].alpha1[Re] = dt*zeta*(4.*mu2*zeta*gamma[pos]*dxgamma[pos] +
                (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dygamma[pos])*den_sq/2.;
            C[pos].alpha1[Im] = dt*zeta*(-4.*mu2*zeta*gamma[pos]*dygamma[pos] +
                    (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dxgamma[pos])*den_sq/2.;
            
            C[pos].beta1[Re] = -2.*j21*C[pos].alpha1[Re];
            C[pos].beta1[Im] = -2.*j21*C[pos].alpha1[Im];
        }
#else
    double mu[2] = {0.,0.};
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            double cosg = expgamma[Re + 2*pos], sing = expgamma[Im + 2*pos];
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            
            double mur = - zeta*dt*( sing*dxgamma[pos] - cosg*dygamma[pos] )/(2.*mu2);
            double mui = dt*zeta*( cosg*dxgamma[pos] + sing*dygamma[pos] )/(2.*mu2);
            
            C[pos].mu1[Re] = dt*(p[1] - 1. + zeta*zeta/(2.*mu2)) + mur;
            C[pos].mu1[Im] = mui;
            
            mu[Re] += mur; //Need to compensate this part of the linear coefficient
            mu[Im] += mui; // to remove chirality
            
            C[pos].xi = dt*x12*j21/mu2;
            C[pos].D = dt*D0 - dt/(4.*mu2);
            C[pos].kappa1 = dt*x12/(2.*mu2);
            C[pos].kappa2 = dt*j21/mu2;
            
            C[pos].alpha1[Re] = -dt*zeta*cosg/(4.*mu2);
            C[pos].alpha1[Im] = dt*zeta*sing/(4.*mu2);
            
            C[pos].alpha2[Re] = dt*zeta*cosg/(2.*mu2);
            C[pos].alpha2[Im] = dt*zeta*sing/(2.*mu2);
            
            C[pos].beta1[Re] = dt*zeta*j21*cosg/(2.*mu2);
            C[pos].beta1[Im] = -dt*zeta*j21*sing/(2.*mu2);
            
            C[pos].beta2[Re] = -dt*zeta*x12*cosg/mu2;
            C[pos].beta2[Im] = -dt*zeta*x12*sing/mu2;
            
            C[pos].chi[Re] = dt*zeta*cosg/2.;
            C[pos].chi[Im] = dt*zeta*sing/2.;
        }
    
    mu[Re] *= norm;
    mu[Im] *= norm;
    
    for(x=0;x<Nx*Ny;x++)
    {
        C[x].mu1[Re] -= mu[Re];
        C[x].mu1[Im] -= mu[Im];
    }
#endif
    
    //------------ Save quenched --------------//
    
#ifndef RANDOM_TORQUE
    delete[] expgamma;
#endif
    
    delete[] gamma;
    delete[] dxgamma;
    delete[] dygamma;
}

/**
 * @brief cal coeffiecients, two noise strengths are used: eta for collision
 * noise, while eta_sd for self diffusion noise.
 * 
 */
void Coefficients_computation(Equation_coefficients *C, int Nx, int Ny, double Lx, double Ly,
                              double density, double eta, double eta_sd, double D0, double zeta,
                              double dt, int load_quenched, const char *name_quenched)
{
    double Intpos[Nmod+1][Nmod+1], Intneg[Nmod+1][Nmod+1], p[Nmod+1], p_sd[Nmod+1];
    
    for(int k=0;k<=Nmod;k++)
    {
        p[k] = exp(-k*k*eta*eta/2.);
        p_sd[k] = exp(-k * k * eta_sd * eta_sd / 2.);
        for(int q=0;q<=Nmod;q++)
        {
            Intpos[k][q] = Int(k,q);
            Intneg[k][q] = Int(k,-q);
        }
    }
    
    double mu2 = p_sd[2] - 1. + ( p[2]*(Intpos[2][0] + Intpos[2][2]) \
                              - Intpos[0][2] - Intpos[0][0] )*density;
    double x12 = density*(p[1]*(Intpos[1][2] + Intneg[1][1]) \
                          - Intpos[0][2] - Intneg[0][1]);
    double j21 = density*(p[2]*Intpos[2][1] - Intpos[0][1]);
    
    int x,y;
    
    //-------------- Compute/load quench disorder-------------------//
    double norm = 1./(double)(Nx*Ny);
    
    double *gamma = new double[Nx*Ny];
#ifndef RANDOM_TORQUE
    double *expgamma = new double[2*Nx*Ny];
#endif
    
    int Nyh = Ny/2 + 1;
    double *dxgamma = new double[Nx*Ny];
    double *dygamma = new double[Nx*Ny];
    
    fftw_complex *FFT_gamma = fftw_alloc_complex(Nx*Nyh);
    fftw_complex *support = fftw_alloc_complex(Nx*Nyh);
    
    fftw_plan forward_gamma = fftw_plan_dft_r2c_2d(Nx,Ny,gamma,FFT_gamma,FFTW_MEASURE);
    fftw_plan backward_gamma = fftw_plan_dft_c2r_2d(Nx,Ny,FFT_gamma,gamma,FFTW_MEASURE);
    fftw_plan dx_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dxgamma,FFTW_MEASURE);
    fftw_plan dy_plan = fftw_plan_dft_c2r_2d(Nx,Ny,support,dygamma,FFTW_MEASURE);
    
    if(!load_quenched)
    {
#ifdef RANDOM_TORQUE
        double avg_gamma = 0.;
#else
        double avg_field[2] = {0.,0.};
#endif
        int Mx = int(Lx);
        int My = int(Ly);
        int block_size_x = Nx / Mx;
        int block_size_y = Ny / My;
        for (x= 0; x < Mx; x++) {
            for (y = 0; y < My; y++) {
                double tmp = M_PI * (2. * PRNG_double01() - 1.);
                for (int i = 0; i < block_size_x; i++) {
                    for (int j = 0; j < block_size_y; j++) {
                        int pos = (y * block_size_y + j) + Ny * (x * block_size_x + i);
                        gamma[pos] = tmp;
#ifdef RANDOM_TORQUE
                        avg_gamma += gamma[pos];
#else
                        expgamma[Re + 2 * pos] = cos(gamma[pos]);
                        expgamma[Im + 2 * pos] = sin(gamma[pos]);

                        avg_field[Re] += expgamma[Re + 2 * pos];
                        avg_field[Im] += expgamma[Im + 2 * pos]; 
#endif
                    }
                }
            }
        }

#ifdef RANDOM_TORQUE
        avg_gamma *= norm;
        for(x=0;x<Nx*Ny;x++)
            gamma[x] -= avg_gamma;
        
        ofstream out(name_quenched,ios::out|ios::binary);
        out.write((char*)gamma,Nx*Ny*sizeof(double));
        out.close();
#else
        avg_field[Re] *= norm;
        avg_field[Im] *= norm;
        for(x=0;x<Nx*Ny;x++)
        {
            expgamma[Re + 2*x] -= avg_field[Re];
            expgamma[Im + 2*x] -= avg_field[Im];
            gamma[x] = atan2(expgamma[Im + 2*x],expgamma[Re + 2*x]);
        }
        
        ofstream out(name_quenched,ios::out|ios::binary);
        out.write((char*)expgamma,2*Nx*Ny*sizeof(double));
        out.close();
#endif
    } else {
        ifstream in(name_quenched,ios::in|ios::binary);
        if(in.fail())
            cout << "Couldn't load quenched, no file found." << endl;
        else {
#ifdef RANDOM_TORQUE
            in.read((char*)gamma,Nx*Ny*sizeof(double));
            double gamma_m = 0;
            for (int i = 0; i < Nx * Ny; i++) {
                gamma_m += gamma[i];
            }
            std::cout << "mean of gamma: " << gamma_m / (Nx * Ny) << std::endl;
#else
            in.read((char*)expgamma,2*Nx*Ny*sizeof(double));
            for(x=0;x<Nx*Ny;x++)
                gamma[x] = atan2(expgamma[Im + 2*x],expgamma[Re + 2*x]);
#endif
            in.close();
        }
    }
    
    //-------------- Compute gradient of gamma-------------------//
    
    fftw_execute(forward_gamma);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qx[x]*FFT_gamma[pos][1];
            support[pos][1] = qx[x]*FFT_gamma[pos][0];
        }
    fftw_execute(dx_plan);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Nyh;y++)
        {
            int pos = y + Nyh*x;
            support[pos][0] = -qy[y]*FFT_gamma[pos][1];
            support[pos][1] = qy[y]*FFT_gamma[pos][0];
        }
    fftw_execute(dy_plan);
    
    fftw_execute(backward_gamma);
    
    fftw_destroy_plan(forward_gamma);
    fftw_destroy_plan(backward_gamma);
    fftw_destroy_plan(dx_plan);
    fftw_destroy_plan(dy_plan);
    
    fftw_free(FFT_gamma);
    fftw_free(support);
    
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            gamma[pos] *= norm;
            dxgamma[pos] *= norm;
            dygamma[pos] *= norm;
        }
    
    //---------- Compute coefficients according to type of quenched --------------//
    
#ifdef RANDOM_TORQUE
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            C[pos].mu1[Re] = dt*(p_sd[1] - 1.);
            C[pos].mu1[Im] = dt*zeta*gamma[pos];
            
            double den_sq = 1./(mu2*mu2 + 4.*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].xi[Re] = dt*x12*j21*mu2*den_sq;
            C[pos].xi[Im] = -dt*x12*j21*2.*zeta*gamma[pos]*den_sq;
            
            C[pos].D[Re] = dt*D0 - dt*mu2*den_sq/4.;
            C[pos].D[Im] = dt*zeta*gamma[pos]*den_sq/2.;
            
            C[pos].kappa1[Re] = dt*x12*mu2*den_sq/2.;
            C[pos].kappa1[Im] = -dt*x12*zeta*gamma[pos]*den_sq;
            
            C[pos].kappa2[Re] = dt*j21*mu2*den_sq;
            C[pos].kappa2[Im] = -dt*2.*j21*zeta*gamma[pos]*den_sq;
            
            den_sq = 1./( (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*\
                         (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos]) + \
                         16.*mu2*mu2*zeta*zeta*gamma[pos]*gamma[pos]);
            
            C[pos].alpha1[Re] = dt*zeta*(4.*mu2*zeta*gamma[pos]*dxgamma[pos] +
                (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dygamma[pos])*den_sq/2.;
            C[pos].alpha1[Im] = dt*zeta*(-4.*mu2*zeta*gamma[pos]*dygamma[pos] +
                    (mu2*mu2 - 4.*zeta*zeta*gamma[pos]*gamma[pos])*dxgamma[pos])*den_sq/2.;
            
            C[pos].beta1[Re] = -2.*j21*C[pos].alpha1[Re];
            C[pos].beta1[Im] = -2.*j21*C[pos].alpha1[Im];
        }
#else
    double mu[2] = {0.,0.};
    for(x=0;x<Nx;x++)
        for(y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            double cosg = expgamma[Re + 2*pos], sing = expgamma[Im + 2*pos];
            
            C[pos].mu1_rho = dt*density*(p[1]*(Intpos[1][0] + Intpos[1][1]) \
                                         - Intpos[0][1] - Intpos[0][0]);
            
            double mur = - zeta*dt*( sing*dxgamma[pos] - cosg*dygamma[pos] )/(2.*mu2);
            double mui = dt*zeta*( cosg*dxgamma[pos] + sing*dygamma[pos] )/(2.*mu2);
            
            C[pos].mu1[Re] = dt*(p_sd[1] - 1. + zeta*zeta/(2.*mu2)) + mur;
            C[pos].mu1[Im] = mui;
            
            mu[Re] += mur; //Need to compensate this part of the linear coefficient
            mu[Im] += mui; // to remove chirality
            
            C[pos].xi = dt*x12*j21/mu2;
            C[pos].D = dt*D0 - dt/(4.*mu2);
            C[pos].kappa1 = dt*x12/(2.*mu2);
            C[pos].kappa2 = dt*j21/mu2;
            
            C[pos].alpha1[Re] = -dt*zeta*cosg/(4.*mu2);
            C[pos].alpha1[Im] = dt*zeta*sing/(4.*mu2);
            
            C[pos].alpha2[Re] = dt*zeta*cosg/(2.*mu2);
            C[pos].alpha2[Im] = dt*zeta*sing/(2.*mu2);
            
            C[pos].beta1[Re] = dt*zeta*j21*cosg/(2.*mu2);
            C[pos].beta1[Im] = -dt*zeta*j21*sing/(2.*mu2);
            
            C[pos].beta2[Re] = -dt*zeta*x12*cosg/mu2;
            C[pos].beta2[Im] = -dt*zeta*x12*sing/mu2;
            
            C[pos].chi[Re] = dt*zeta*cosg/2.;
            C[pos].chi[Im] = dt*zeta*sing/2.;
        }
    
    mu[Re] *= norm;
    mu[Im] *= norm;
    
    for(x=0;x<Nx*Ny;x++)
    {
        C[x].mu1[Re] -= mu[Re];
        C[x].mu1[Im] -= mu[Im];
    }
#endif
    
    //------------ Save quenched --------------//
    
#ifndef RANDOM_TORQUE
    delete[] expgamma;
#endif
    
    delete[] gamma;
    delete[] dxgamma;
    delete[] dygamma;
}