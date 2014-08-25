/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Hartwig Anzt

       Utilities for testing MAGMA-sparse.
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/file.h>
#include <errno.h>
#include <sys/stat.h>
// includes, project
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"


// --------------------
const char *usage_sparse_short =
"Usage: %s [options] [-h|--help]\n\n";

const char *usage_sparse =
"Options are:\n"
"  --range start:stop:step\n"
"                   Adds test cases with range for sizes m,n,k. Can be repeated.\n"
"  -N m[,n[,k]]     Adds one test case with sizes m,n,k. Can be repeated.\n"
"                   If only m,n given then k=n. If only m given then n=k=m.\n"
"  -m m             Sets m for all tests, overriding -N and --range.\n"
"  -n n             Sets n for all tests, overriding -N and --range.\n"
"  -k k             Sets k for all tests, overriding -N and --range.\n"
"  Default test sizes are the range 1088 : 10304 : 1024, that is, 1K+64 : 10K+64 : 1K.\n"
"\n"
"  -c  --[no]check  Whether to check results. Some tests always check.\n"
"                   Also set with $MAGMA_TESTINGS_CHECK.\n"
"  -c2 --check2     For getrf, check residual |Ax-b| instead of |PA-LU|.\n"
"  -l  --[no]lapack Whether to run lapack. Some tests always run lapack.\n"
"                   Also set with $MAGMA_RUN_LAPACK.\n"
"      --[no]warmup Whether to warmup. Not yet implemented in most cases.\n"
"                   Also set with $MAGMA_WARMUP.\n"
"      --[not]all   Whether to test all combinations of flags, e.g., jobu.\n"
"  --dev x          GPU device to use, default 0.\n"
"  --pad n          Pad LDDA on GPU to multiple of pad, default 32.\n"
"  --verbose        Verbose output.\n"
"  -x  --exclusive  Lock file for exclusive use (internal ICL functionality).\n"
"\n"
"The following options apply to only some routines.\n"
"  --nb x           Block size, default set automatically.\n"
"  --nrhs x         Number of right hand sides, default 1.\n"
"  --nstream x      Number of CUDA streams, default 1.\n"
"  --ngpu x         Number of GPUs, default 1. Also set with $MAGMA_NUM_GPUS.\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  --nthread x      Number of CPU threads, default 1.\n"
"  --itype [123]    Generalized Hermitian-definite eigenproblem type, default 1.\n"
"  --svd_work [0123] SVD workspace size, from min (1) to optimal (3), or query (0), default 0.\n"
"  --version x      version of routine, e.g., during development, default 1.\n"
"  --fraction x     fraction of eigenvectors to compute, default 1.\n"
"  --tolerance x    accuracy tolerance, multiplied by machine epsilon, default 30.\n"
"  --panel_nthread x Number of threads in the first dimension if the panel is decomposed into a 2D layout, default 1.\n"
"  --fraction_dcpu x Percentage of the workload to schedule on the cpu. Used in magma_amc algorithms only, default 0.\n"
"  -L -U -F         uplo   = Lower*, Upper, or Full.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -[TC]            transA = Trans or ConjTrans. Default is NoTrans. Doesn't change transB.\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"  -U[NASO]         jobu   = No*, All, Some, or Overwrite; compute left  singular vectors. gesdd uses this for jobz.\n"
"  -V[NASO]         jobvt  = No*, All, Some, or Overwrite; compute right singular vectors.\n"
"  -J[NV]           jobz   = No* or Vectors; compute eigenvectors (symmetric).\n"
"  -L[NV]           jobvl  = No* or Vectors; compute left  eigenvectors (non-symmetric).\n"
"  -R[NV]           jobvr  = No* or Vectors; compute right eigenvectors (non-symmetric).\n"
"                   * default values\n";

extern "C"
magma_int_t
magma_zparse_opts( int argc, char** argv, magma_zopts *opts, int *matrices )
{
    // negative flag indicating -m, -n, -k not given
    int m = -1;
    int n = -1;
    int k = -1;
    
    // fill in default values
    opts->input_format = Magma_CSR;
    opts->blocksize = 8;
    opts->alignment = 8;
    opts->output_format = Magma_CSR;
    opts->input_location = Magma_CPU;
    opts->output_location = Magma_CPU;
    opts->scaling = Magma_NOSCALE;
    opts->solver_par.epsilon = 10e-16;
    opts->solver_par.maxiter = 1000;
    opts->solver_par.verbose = 0;
    opts->solver_par.version = 0;
    opts->solver_par.num_eigenvalues = 0;
    opts->precond_par.solver = Magma_JACOBI;
    
    printf( usage_sparse_short, argv[0] );
    
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    
    int info;
    int ntest = 0;
    

    for( int i = 1; i < argc; ++i ) {
     if ( strcmp("--format", argv[i]) == 0 ) {
            info = atoi( argv[++i] );
            switch( info ) {
                case 0: opts->output_format = Magma_CSR; break;
                case 1: opts->output_format = Magma_ELL; break;
                case 2: opts->output_format = Magma_ELLRT; break;
                case 3: opts->output_format = Magma_SELLP; break;
            }
        }else if ( strcmp("--mscale", argv[i]) == 0 ) {
            info = atoi( argv[++i] );
            switch( info ) {
                case 0: opts->scaling = Magma_NOSCALE; break;
                case 1: opts->scaling = Magma_UNITDIAG; break;
                case 2: opts->scaling = Magma_UNITROW; break;
            }

        }else if ( strcmp("--solver", argv[i]) == 0 ) {
            info = atoi( argv[++i] );
            switch( info ) {
                case 0: opts->solver_par.solver = Magma_CG; break;
                case 1: opts->solver_par.solver = Magma_CGMERGE; break;
                case 2: opts->solver_par.solver = Magma_BICGSTAB; break;
                case 3: opts->solver_par.solver = Magma_BICGSTABMERGE; break;
                case 4: opts->solver_par.solver = Magma_JACOBI; break;
                case 5: opts->solver_par.solver = Magma_GMRES; break;
                case 6: opts->solver_par.solver = Magma_ITERREF; break;
                case 7: opts->solver_par.solver = Magma_LOBPCG; break;
                case 8: opts->solver_par.solver = Magma_PCG; break;
                case 9: opts->solver_par.solver = Magma_PBICGSTAB; break;
                case 10: opts->solver_par.solver = Magma_PGMRES; break;
                case 11: opts->solver_par.solver = Magma_BAITER; break;
            }
        }else if ( strcmp("--precond", argv[i]) == 0 ) {
            info = atoi( argv[++i] );
            switch( info ) {
                case 0: opts->precond_par.solver = Magma_NONE; break;
                case 1: opts->precond_par.solver = Magma_JACOBI; break;
                case 2: opts->precond_par.solver = Magma_ILU; break;

            }
        }else if ( strcmp("--blocksize", argv[i]) == 0 ) {
            opts->blocksize = atoi( argv[++i] );
        }else if ( strcmp("--alignment", argv[i]) == 0 ) {
            opts->alignment = atoi( argv[++i] );
        }else if ( strcmp("--verbose", argv[i]) == 0 ) {
            opts->solver_par.verbose = atoi( argv[++i] );
        }  else if ( strcmp("--maxiter", argv[i]) == 0 ) {
            opts->solver_par.maxiter = atoi( argv[++i] );
        } else if ( strcmp("--tol", argv[i]) == 0 ) {
            sscanf( argv[++i], "%lf", &opts->solver_par.epsilon );
        } else if ( strcmp("--eigenvalues", argv[i]) == 0 ) {
            opts->solver_par.num_eigenvalues = atoi( argv[++i] );
        } else if ( strcmp("--version", argv[i]) == 0 ) {
            opts->solver_par.version = atoi( argv[++i] );
        }        
        // ----- usage
        else if ( strcmp("-h",     argv[i]) == 0 ||
                  strcmp("--help", argv[i]) == 0 ) {
            fprintf( stderr, usage_sparse, argv[0] );
            exit(0);
        } else{
            *matrices = i;
            break;
        }
    }
    return MAGMA_SUCCESS;

}

    

