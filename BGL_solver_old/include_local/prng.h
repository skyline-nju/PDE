/*PRNG*/

// You should *always* seed the RNG.  The usual time to do it is the
// point in time when you create RNG (typically at the beginning of the
// program).
//
// pcg32_srandom_r takes two 64-bit constants (the initial state, and the
// rng sequence selector; rngs with different sequence selectors will
// *never* have random sequences that coincide, at all) -

// Seed with external entropy -- the time and some program addresses
// (which will actually be somewhat random on most modern systems).

void PRNG_random_seed()
{
    int r = Nmod;
    
    pcg32_srandom_r(&rng, time(NULL) ^ (intptr_t)&printf,(intptr_t)&r);
}

void PRNG_random_seed(unsigned long long seed) {
    int r = Nmod;
    pcg32_srandom_r(&rng, (intptr_t)&seed, (intptr_t)&r);
}

void PRNG_deterministic_seed(uint64_t initstate, uint64_t initseq)
{
    pcg32_srandom_r(&rng, initstate, initseq);
}

static inline double PRNG_double01()
{
    return (double)pcg32_random_r(&rng)/(double)UINT32_MAX;
}

static inline void PRNG_gaussian01(double *random)
{
    double u1, u2, r = 0.;
    
    while(r==0 || r >= 1)
    {
        u1 = 2.*PRNG_double01() - 1.;
        u2 = 2.*PRNG_double01() - 1.;
        r = u1*u1 + u2*u2;
    }
    
    random[0] = u1*sqrt(-2.*log(r)/r);
    random[1] = u2*sqrt(-2.*log(r)/r);
}
