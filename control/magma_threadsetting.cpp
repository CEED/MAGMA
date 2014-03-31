/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Simplice Donfack
*/
#include "common_magma.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(MAGMA_WITH_MKL)
#include <mkl_service.h>
#endif

/***************************************************************************//**
 * switch lapack thread_num initialization
 **/

//=============================================================
// Determine the number of threads by order checking OMP then 
// MAGMA_NUM_THREADS ten the system otherwise it returns 1
//=============================================================
magma_int_t magma_get_parallel_numthreads()
{
    /* determine the number of threads */
    magma_int_t threads = 1, mthreads=0;

    // First check OMP_NUM_THREADS then the system CPUs
#if defined(_OPENMP)
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
#else
    #ifdef _MSC_VER  // Windows
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );
    ncores = sysinfo.dwNumberOfProcessors;
    #else
    ncores = sysconf(_SC_NPROCESSORS_ONLN);
    #endif
    /*
    const char *input_threads_str = getenv("MAGMA_NUM_THREADS");
    magma_int_t threads = ncores;
    if ( input_threads_str != NULL ) {
        char* endptr;
        threads = strtol( input_threads_str, &endptr, 10 );
        if ( threads < 1 || *endptr != '\0' ) {
            threads = 1;
            fprintf( stderr, "$MAGMA_NUM_THREADS=%s is an invalid number; using %d threads.\n",
                     input_threads_str, (int) threads );
        }
        else if ( threads > MagmaMaxGPUs || threads > ndevices ) {
            threads = min( ndevices, MagmaMaxGPUs );
            fprintf( stderr, "$MAGMA_NUM_THREADS=%s exceeds available CPUs=%d; using %d CPUs.\n",
                     input_threads_str, ncores, (int) threads );
        }
        assert( 1 <= threads && threads <= ncores );
    }
    */
    const char* myenv = getenv("MAGMA_NUM_THREADS");
    if (myenv != NULL) {
        mthreads = atoi(myenv);
        if (mthreads < threads) {
            threads = mthreads;
        }
    }
#endif

    // Fourth use one thread
    if (threads < 1)
        threads = 1;

    return threads;
}
//=============================================================


//=============================================================
// determine the BLAS/LAPACK number of threads by checking first
// for MKL then OMP otherwise return 1
//=============================================================
magma_int_t magma_get_lapack_numthreads()
{
    /* determine the number of threads */
    magma_int_t threads = 1;

    // First check OMP_NUM_THREADS then MKL then the system CPUs
#if defined(MAGMA_WITH_MKL)
    threads = mkl_get_max_threads();
#elif defined(_OPENMP)
    #pragma omp parallel
    {
        threads = omp_get_num_threads();
    }
#endif

    return threads;
}
//=============================================================

//=============================================================
void magma_set_lapack_numthreads(magma_int_t num_threads)
{
#if defined(MAGMA_WITH_MKL)
    mkl_set_num_threads( num_threads );
#elif defined(_OPENMP)
    omp_set_num_threads( num_threads );
#endif
}
//=============================================================





/*
 #else
    #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "Also, if using multi-threaded MKL, please compile MAGMA with -DMAGMA_WITH_MKL.\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
    #endif
   #ifndef NO_WARNING
    printf("==============================================================================\n"
           "WARNING: a parallel section of MAGMA could not be run in parallel because\n"
           "OpenMP was not enabled; add -fopenmp (for gcc) or -openmp (for icc) to\n"
           "both compilation and linkage flags (OPTS and LDOPTS in make.inc).\n"
           "To disable this warning, compile MAGMA with -DNO_WARNING.\n"
           "==============================================================================\n");
    #endif

*/

