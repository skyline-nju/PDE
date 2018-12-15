
/****************************Observables and saving***********************/

void Order(const int N, const double *f, double *obs)
{
    for(int k=0;k<Nfields;k++)
    {
        obs[k] = 0.;
        
        for(int x=0;x<N;x++)
            obs[k] += f[k + Nfields*x];
        
        obs[k] /= (double)N;
    }
}

void Save_obs(const int N, const double t, const double *f, string name, \
              const double dt, const int append)
{
    double order[Nfields];
   
    Order(N,f,order);
    
    ofstream out;
    if(append)
        out.open(name.c_str(),ios::out|ios::app);
    else
        out.open(name.c_str(),ios::out);
    
    out << t << " ";
    for(int k=0;k<Nfields;k++)
        out << order[k] << " ";
    out << endl;
    
    cout << t << " " << order[rho] << " ";
    cout << sqrt(order[px]*order[px] + order[py]*order[py]) << " ";
    cout << atan2(order[py],order[px]) << endl;
    
    out.close();
}

void Save_bin(const int Nx, const int Ny, const double dx, const double dy, \
              const double *f, string name)
{
    int N = Nx*Ny;
    double **Store_array = new double*[N];
    
    for(int i=0;i<N;i++)
        Store_array[i] = new double[Nfields + 2];
    
    for(int x=0;x<Nx;x++)
        for(int y=0;y<Ny;y++)
        {
            int pos = (y + Ny*x);
            
            Store_array[pos][0] = x*dx;
            Store_array[pos][1] = y*dy;
            
            for(int k=0;k<Nfields;k++)
                Store_array[pos][k+2] = f[k + Nfields*pos];
        }
    
    ofstream out(name.c_str(),ios::out|ios::binary);
    for(int i=0;i<N;i++)
        out.write((char*)Store_array[i],(Nfields+2)*sizeof(double));
    out.close();
    
    for(int i=0;i<N;i++)
        delete[] Store_array[i];
    delete[] Store_array;
}
