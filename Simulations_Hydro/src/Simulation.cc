#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <cstring>

#include "output.h"
/*-------------------------------------------------------------------------------
 
Nonlinear pseudo spectral PDE solver of the hydrodynamic equations for the polar class
 with random torques (RT) or random field (RF).
 Fourier transforms are computed using the fftw library: http://www.fftw.org .
 There are two choices of temporal schemes: Euler (sheme = 1) of 4th order Runge Kutta(scheme = 2).
 The random number generator is PCG: http://www.pcg-random.org .
 The compilation is simply done with the associated Makefile, RT and RF are compiled separately.
 
 The program can be run with parameters imported from the file "parameters.dat" or using
 the command line (see Function Command_line() below for all entries).
 Example: ./SimRT -density 1 -zeta 0.2 -scheme 1 -load_quenched etc...
 Default parameters are given at the beginning of the main.
 
-------------------------------------------------------------------------------*/

// FFT library
#include <fftw3.h>

//Random number generator (pcg32)
#include "../include_random/pcg_basic.h"
pcg32_random_t rng;

using namespace std;

//Definitions
enum {Re = 0, Im};
enum {rho = 0, px, py, Nfields}; //Density, and polar fields
#define Nmod 3

struct Equation_coefficients
{
#ifdef RANDOM_TORQUE
    double mu1[2];
    double mu1_rho;
    double xi[2];
    double D[2];
    double kappa1[2], kappa2[2];
    double alpha1[2], beta1[2];
#else
    double mu1[2], mu1_rho, xi, D, kappa1, kappa2;
    double alpha1[2], alpha2[2], beta1[2], beta2[2], chi[2];
#endif
};

/*---------------------------Simulation parameters---------------------------*/

//Lattice size
int Nx, Ny, N;

//Time related stuffs
int Nstep, Nsave_obs, Nsave_bin;
double dt, phys_time;

//System size
double Lx, Ly, L, dx, dy;
double *qx, *qy; //Frenquencies

//Parameters of the model
double density, eta;
double D0; //Additional diffusion (suppress spurious instability)
double noise_ic;

//Quenched disorder amplitude
double zeta;
int load_quenched;
char name_quenched[256];

//Coefficients
Equation_coefficients *C;

//Arrays
int alloc_real, alloc_complex;
double *f, *dxf, *dyf, *Nonlinear;
#ifdef RANDOM_TORQUE
double *d2xf, *d2yf;
#endif
fftw_complex *FFT_f, *FFT_Linear, *FFT_Nonlinear;
fftw_complex *FFT_support, *FFT_support_derivatives, *FFT_beta;

// plans of the FFTW
fftw_plan forward_plan, backward_plan, backward_dx_plan, backward_dy_plan;
fftw_plan forward_NL_plan, backward_plan_beta;

#ifdef RANDOM_TORQUE
fftw_plan backward_d2x_plan, backward_d2y_plan;
#endif

//File stuffs
char nameic[256], name_file_IC[256], nameobs[256], namebin[256], Q[128], namefile[256];
ifstream param;

//Others
int scheme, do_antialiasing;
double fftw_norm;

#include "../include_local/prng.h"
#include "../include_local/Coeffs.h"
#include "../include_local/Dyn.h"
#include "../include_local/ic.h"
// #include "../include_local/io.h"

//Reads parameters from a file
void Load_config_file()
{
    param.open("../parameters.dat",ios::in);
    param >> Nstep >> Nsave_obs >> Nsave_bin >> density >> eta >> zeta >> D0\
    >> Lx >> Ly >> Nx >> Ny >> dt >> scheme >> nameic >> namefile;
    param.close();
}

void Command_line(int arg, char *args[])
{
    int Argument_cmd=1;
    while (Argument_cmd<arg)
    {
        if(strcmp(args[Argument_cmd],"-density")==0)
        {
            Argument_cmd += 1;
            density = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-eta")==0)
        {
            Argument_cmd += 1;
            eta = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-D0")==0)
        {
            Argument_cmd += 1;
            D0 = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-zeta")==0)
        {
            Argument_cmd += 1;
            zeta = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Lx")==0)
        {
            Argument_cmd += 1;
            Lx = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Ly")==0)
        {
            Argument_cmd += 1;
            Ly = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-L")==0)
        {
            Argument_cmd += 1;
            Lx = atoi(args[Argument_cmd]);
            Ly = Lx;
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Nx")==0)
        {
            Argument_cmd += 1;
            Nx = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Ny")==0)
        {
            Argument_cmd += 1;
            Ny = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-N")==0)
        {
            Argument_cmd += 1;
            Nx = atoi(args[Argument_cmd]);
            Ny = Nx;
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-dt")==0)
        {
            Argument_cmd += 1;
            dt = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-do_antialiasing")==0)
        {
            Argument_cmd += 1;
            do_antialiasing = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-type_ic")==0)
        {
            Argument_cmd += 1;
            strcpy(nameic,args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-noise_ic")==0)
        {
            Argument_cmd += 1;
            noise_ic = atof(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-name_file_ic")==0)
        {
            Argument_cmd += 1;
            strcpy(name_file_IC,args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Name")==0)
        {
            Argument_cmd += 1;
            strcpy(namefile,args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Nstep")==0)
        {
            Argument_cmd += 1;
            Nstep = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Nsaveobs")==0)
        {
            Argument_cmd += 1;
            Nsave_obs = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-Nsavebin")==0)
        {
            Argument_cmd += 1;
            Nsave_bin = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-scheme")==0)
        {
            Argument_cmd += 1;
            scheme = atoi(args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else if(strcmp(args[Argument_cmd],"-load_quenched")==0)
        {
            Argument_cmd += 1;
            load_quenched = 1;
        }
        else if(strcmp(args[Argument_cmd],"-name_quenched")==0)
        {
            Argument_cmd += 1;
            strcpy(name_quenched,args[Argument_cmd]);
            Argument_cmd += 1;
        }
        else
            Argument_cmd++;
    }
}

void Set_parameters()
{
    N = Nx*Ny;
    L = Lx*Ly;
    
    dx=Lx/(double)Nx;
    dy=Ly/(double)Ny;
    fftw_norm = 1./(double)N;
    
    alloc_complex = Nfields*Nx*(Ny/2+1);
    alloc_real = Nfields*N;
    
    //Frequencies
    qx = new double[Nx];
    qy = new double[Ny/2+1];
    double dkx=2.*M_PI/Lx, dky = 2.*M_PI/Ly;
    
    for(int x=0;x<Nx;x++)
    {
        if(x<=Nx/2)
            qx[x] = dkx*x;
        else
            qx[x] = dkx*(x - Nx);
    }
    
    for(int i=0;i<=Ny/2;i++)
        qy[i] = dky*i;
}

void Allocate_arrays()
{
    C = (Equation_coefficients*)malloc(N*sizeof(Equation_coefficients));
    
    f = fftw_alloc_real(alloc_real); // Density and polar fields array
    
    FFT_f = fftw_alloc_complex(alloc_complex);
    FFT_Linear = fftw_alloc_complex(alloc_complex);
    
    dxf = fftw_alloc_real(alloc_real);
    dyf = fftw_alloc_real(alloc_real);
    
#ifdef RANDOM_TORQUE
    d2xf = fftw_alloc_real(alloc_real);
    d2yf = fftw_alloc_real(alloc_real);
#endif
    
    Nonlinear = fftw_alloc_real(alloc_real);
    FFT_Nonlinear = fftw_alloc_complex(alloc_complex);
    
    FFT_support = fftw_alloc_complex(alloc_complex);
    FFT_support_derivatives = fftw_alloc_complex(alloc_complex);
    
    int n[] = {Nx,Ny};
    
    if(scheme == 2)
    {
        FFT_beta = fftw_alloc_complex(alloc_complex);
        backward_plan_beta = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_beta,NULL,Nfields,\
                                                    1,f,NULL,Nfields,1,FFTW_MEASURE);
    }
    
    forward_plan = fftw_plan_many_dft_r2c(2,n,Nfields,f,NULL,Nfields,1,FFT_f,NULL,Nfields,\
                                          1,FFTW_MEASURE);
    
    backward_plan = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_f,NULL,Nfields,1,f,NULL,Nfields,\
                                           1,FFTW_MEASURE);
    
    backward_dx_plan = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_support_derivatives,NULL,Nfields,\
                                              1,dxf,NULL,Nfields,1,FFTW_MEASURE);
    backward_dy_plan = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_support_derivatives,NULL,Nfields,\
                                              1,dyf,NULL,Nfields,1,FFTW_MEASURE);
    
#ifdef RANDOM_TORQUE
    backward_d2x_plan = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_support_derivatives,NULL,Nfields,\
                                               1,d2xf,NULL,Nfields,1,FFTW_MEASURE);
    backward_d2y_plan = fftw_plan_many_dft_c2r(2,n,Nfields,FFT_support_derivatives,NULL,Nfields,\
                                               1,d2yf,NULL,Nfields,1,FFTW_MEASURE);
#endif
    
    forward_NL_plan = fftw_plan_many_dft_r2c(2,n,Nfields,Nonlinear,NULL,Nfields,\
                                             1,FFT_Nonlinear,NULL,Nfields,1,FFTW_MEASURE);
}

/****************************Main***********************/

int main(int arg, char *args[])
{
    // PRNG_random_seed();
    PRNG_random_seed(1);
    //Default parameters of the simulation
    density = 1.;
    eta = 0.8;
    zeta = 0.;
    Lx = 128;
    Ly = Lx;
    Nstep = 1E5;
    Nsave_obs = 1E2;
    Nsave_bin = 1E3;
    Nx = 64;
    Ny = Nx;
    
    noise_ic = 1.e-3;
    dt = 1.e-2;
    
    phys_time = 0.;
    
    scheme = 1;
    do_antialiasing = 1;
    
    load_quenched = 0;
#ifdef RANDOM_TORQUE
    sprintf(name_quenched,"../quenched_realizations/RT/default.bin");
#else
    sprintf(name_quenched,"../quenched_realizations/RF/default.bin");
#endif
    
    Load_config_file();
    Command_line(arg,args);
    
    Set_parameters();
    Allocate_arrays();
    Coefficients_computation(C, Nx, Ny, Lx, Ly, density, eta, D0,
                             zeta, dt, load_quenched, name_quenched);
    // Coefficients_computation(C, Nx, Ny, density, eta, D0,
    //                          zeta, dt, load_quenched, name_quenched);
#ifdef RANDOM_TORQUE
    sprintf(Q,"RT_%s_d%1.2lf_n%1.2lf_qd%1.2lf_Lx%1.0f_Ly%1.0f_Nx%d_Ny%d",\
            namefile,density,eta,zeta,Lx,Ly,Nx,Ny);
#else
    sprintf(Q,"RF_%s_d%1.2lf_n%1.2lf_qd%1.2lf_Lx%1.0f_Ly%1.0f_Nx%d_Ny%d",\
            namefile,density,eta,zeta,Lx,Ly,Nx,Ny);
#endif
    sprintf(nameobs,"../data/obs/obs%s.dat",Q);
    
    get_IC();
    
    //***********************Dynamics*********************************//
    
    cout << "Starting simulation with parameters:" << endl;
    cout << "density=" << density << "              noise=" << eta << endl;
#ifdef RANDOM_TORQUE
    cout << "Quenched disorder (Random torques)=" << zeta << endl;
#else
    cout << "Quenched disorder (Random field)=" << zeta << endl;
#endif
    cout << "Lx=" << Lx << "              Ly=" << Ly << endl;
    cout << "Nx=" << Nx << "              Ny=" << Ny << endl;
    cout << "dt=" << dt << endl;
    cout << "Number of steps=" << Nstep << " and " << nameic << " initial condition" << endl;
    cout << endl;
    
    cout << "time | avg density | magnetization | avg orientation | min density | elapsed time" << endl;
    
    // Save_obs(N,0.,f,nameobs,dt,0);
    auto real_ini_time = std::chrono::system_clock::now();
    save_order_para(f, Nfields, N, 0., nameobs, 0, real_ini_time);
    // sprintf(namebin,"../data/profiles/fields%s_t0.bin",Q);
    // Save_bin(Nx,Ny,dx,dy,f,namebin);
    sprintf(namebin, "../data/profiles/fileds%s.bin", Q);
    save_fields(f, Nfields, Nx, Ny, phys_time, namebin, 0);
    
    fftw_execute(forward_plan);
    Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_f,do_antialiasing);
    
    for(int t=1;t<=Nstep;t++)
    {
        phys_time += dt;
    
        //RK4 (see https://en.wikipedia.org/wiki/Runge–Kutta_methods if needed)
        //please check first Euler routine that is simpler
        if(scheme == 2)
        {
            Field_cpy(alloc_complex,FFT_f,FFT_support);
            
            //Beta1
            Dx(Nx,Ny,FFT_f,FFT_support_derivatives,qx);
            fftw_execute(backward_dx_plan);
            Dy(Nx,Ny,FFT_f,FFT_support_derivatives,qy);
            fftw_execute(backward_dy_plan);
#ifdef RANDOM_TORQUE
            D2x(Nx,Ny,FFT_f,FFT_support_derivatives,qx);
            fftw_execute(backward_d2x_plan);
            D2y(Nx,Ny,FFT_f,FFT_support_derivatives,qy);
            fftw_execute(backward_d2y_plan);
#endif
            
            Linear_part(FFT_f,FFT_Linear,Nx,Ny,C,qx,qy,dt,0);
            fftw_execute(backward_plan);
#ifdef RANDOM_TORQUE
            Nonlinear_part(f,dxf,dyf,d2xf,d2yf,Nonlinear,C);
#else
            Nonlinear_part(f,dxf,dyf,Nonlinear,C);
#endif
            fftw_execute(forward_NL_plan);
            Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_Nonlinear,do_antialiasing);
            
            RK4_scheme(alloc_complex,FFT_f,FFT_Linear,FFT_Nonlinear,\
                       FFT_support,FFT_beta,0,1./6,1,0.5);
            
            //Beta2
            Dx(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_dx_plan);
            Dy(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_dy_plan);
#ifdef RANDOM_TORQUE
            D2x(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_d2x_plan);
            D2y(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_d2y_plan);
#endif
            
            Linear_part(FFT_beta,FFT_Linear,Nx,Ny,C,qx,qy,dt,0);
            fftw_execute(backward_plan_beta);
#ifdef RANDOM_TORQUE
            Nonlinear_part(f,dxf,dyf,d2xf,d2yf,Nonlinear,C);
#else
            Nonlinear_part(f,dxf,dyf,Nonlinear,C);
#endif
            fftw_execute(forward_NL_plan);
            Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_Nonlinear,do_antialiasing);
            
            RK4_scheme(alloc_complex,FFT_f,FFT_Linear,FFT_Nonlinear,\
                       FFT_support,FFT_beta,1,2./6,1,0.5);
            
            //Beta3
            Dx(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_dx_plan);
            Dy(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_dy_plan);
#ifdef RANDOM_TORQUE
            D2x(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_d2x_plan);
            D2y(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_d2y_plan);
#endif
            
            Linear_part(FFT_beta,FFT_Linear,Nx,Ny,C,qx,qy,dt,0);
            fftw_execute(backward_plan_beta);
#ifdef RANDOM_TORQUE
            Nonlinear_part(f,dxf,dyf,d2xf,d2yf,Nonlinear,C);
#else
            Nonlinear_part(f,dxf,dyf,Nonlinear,C);
#endif
            fftw_execute(forward_NL_plan);
            Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_Nonlinear,do_antialiasing);
            
            RK4_scheme(alloc_complex,FFT_f,FFT_Linear,FFT_Nonlinear,\
                       FFT_support,FFT_beta,1,2./6,1,1.);
            
            //Beta4
            Dx(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_dx_plan);
            Dy(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_dy_plan);
#ifdef RANDOM_TORQUE
            D2x(Nx,Ny,FFT_beta,FFT_support_derivatives,qx);
            fftw_execute(backward_d2x_plan);
            D2y(Nx,Ny,FFT_beta,FFT_support_derivatives,qy);
            fftw_execute(backward_d2y_plan);
#endif
            
            Linear_part(FFT_beta,FFT_Linear,Nx,Ny,C,qx,qy,dt,0);
            fftw_execute(backward_plan_beta);
#ifdef RANDOM_TORQUE
            Nonlinear_part(f,dxf,dyf,d2xf,d2yf,Nonlinear,C);
#else
            Nonlinear_part(f,dxf,dyf,Nonlinear,C);
#endif
            fftw_execute(forward_NL_plan);
            Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_Nonlinear,do_antialiasing);
            
            RK4_scheme(alloc_complex,FFT_f,FFT_Linear,FFT_Nonlinear,\
                             FFT_support,FFT_beta,1,1./6,0,1.);
        }
        
        else //Euler
        {
            //Compute space derivatives and put them in real space
            Dx(Nx,Ny,FFT_f,FFT_support_derivatives,qx);
            fftw_execute(backward_dx_plan);
            Dy(Nx,Ny,FFT_f,FFT_support_derivatives,qy);
            fftw_execute(backward_dy_plan);
#ifdef RANDOM_TORQUE
            D2x(Nx,Ny,FFT_f,FFT_support_derivatives,qx);
            fftw_execute(backward_d2x_plan);
            D2y(Nx,Ny,FFT_f,FFT_support_derivatives,qy);
            fftw_execute(backward_d2y_plan);
#endif
            Linear_part(FFT_f,FFT_Linear,Nx,Ny,C,qx,qy,dt,1); //Linear part (in Fourier)
            fftw_execute(backward_plan);
            //Nonlinear part (in real space)
#ifdef RANDOM_TORQUE
            Nonlinear_part(f,dxf,dyf,d2xf,d2yf,Nonlinear,C);
#else
            Nonlinear_part(f,dxf,dyf,Nonlinear,C);
#endif
            fftw_execute(forward_NL_plan);
            Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_Nonlinear,do_antialiasing);
            
            Euler_scheme(alloc_complex,FFT_f,FFT_Linear,FFT_Nonlinear); //Sum all
        }
        
        if(t%Nsave_obs == 0 && t>0)
        {
            fftw_execute(backward_plan);
            
            // Save_obs(N,phys_time,f,nameobs,dt,1);
            save_order_para(f, Nfields, N, phys_time, nameobs, 1, real_ini_time);
            
            if(scheme)
            {
                fftw_execute(forward_plan);
                Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_f,do_antialiasing);
            }
        }
        
        if(t%Nsave_bin == 0 && t>0)
        {
            fftw_execute(backward_plan);
            
            // sprintf(namebin,"../data/profiles/fields%s_t%1.0f.bin",Q,round(phys_time));
            // Save_bin(Nx,Ny,dx,dy,f,namebin);
            save_fields(f, Nfields, Nx, Ny, phys_time, namebin, 1);

            
            if(scheme)
            {
                fftw_execute(forward_plan);
                Antialiasing_normalization(Nx,Ny,fftw_norm,FFT_f,do_antialiasing);
            }
        }
    }
    
    delete[] qx;
    delete[] qy;
    
    fftw_cleanup();
    
    return 0;
}
