/****************************Initial conditions***********************/

void Disordered_IC(const int N, double *f, const double noise_ic)
{
    for(int x=0;x<N;x++)
    {
        f[rho + Nfields*x] = 1.;
        
        for(int k=1;k<Nfields;k++)
            f[k + Nfields*x] = noise_ic*(2.*PRNG_double01() - 1.);
    }
}

void Ordered_IC(const int N, double *f, const Equation_coefficients *C, const double noise_ic)
{
    for(int x=0;x<N;x++)
    {
        f[rho + Nfields*x] = 1.;
        
#ifdef RANDOM_TORQUE
        f[px + Nfields*x] = sqrt((C[0].mu1[Re] + C[0].mu1_rho)/C[0].xi[Re])\
                                    *(1. + noise_ic*(2.*PRNG_double01() - 1.));
#else
        double mu2 = exp(-eta*eta*2.) - 1. + (exp(-eta*eta*2.)*(Int(2,0)+Int(2,2))-Int(0,0)-Int(0,2))*density;
        f[px + Nfields*x] = sqrt((dt*( exp(-eta*eta/2.) - 1. + zeta*zeta/(2.*mu2)) \
                                  + C[0].mu1_rho)/C[0].xi)\
                                    *(1. + noise_ic*(2.*PRNG_double01() - 1.));
#endif
        f[py + Nfields*x] = noise_ic*(2.*PRNG_double01() - 1.);
    }
}

void File_IC(const int Nx, const int Ny, double *f, const char *name, \
             const double noise_ic)
{
    int N = Nx*Ny;
    double **Read_array = new double*[N];
    
    for(int i=0;i<N;i++)
        Read_array[i] = new double[Nfields + 2];
    
    ifstream in(name,ios::in|ios::binary);
    for(int i=0;i<N;i++)
        in.read((char*)Read_array[i],(Nfields+2)*sizeof(double));
    in.close();
    
    for(int x=0;x<Nx;x++)
        for(int y=0;y<Ny;y++)
        {
            int pos = y + Ny*x;
            
            f[rho + Nfields*pos] = Read_array[pos][2];
            
            for(int k=1;k<Nfields;k++)
                f[k + Nfields*pos] = Read_array[pos][k+2]*(1.+noise_ic*(2.*PRNG_double01() - 1.));
        }

    for(int i=0;i<N;i++)
        delete[] Read_array[i];
    delete[] Read_array;
}

void get_IC()
{
    if(strcmp(nameic,"file")==0)
    {
        File_IC(Nx,Ny,f,name_file_IC,noise_ic);
    }
    else if(strcmp(nameic,"polar")==0)
    {
        Ordered_IC(N,f,C,noise_ic);
    }
    else
        Disordered_IC(N,f,noise_ic);
}









