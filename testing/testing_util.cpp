/*
 *  -- MAGMA (version 1.1) --
 *     Univ. of Tennessee, Knoxville
 *     Univ. of California, Berkeley
 *     Univ. of Colorado, Denver
 *     November 2011
 *
 * @precisions normal z -> c s d
 *
 * Utilities for testing.
 * @author Mark Gates
 **/

#include <string.h>
#include <assert.h>

#include "testings.h"

// --------------------
// If condition is false, print error message and exit.
// Error message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        exit(1);
    }
}


// --------------------
const char *usage_short =
"Usage: %s [options] [-h|--help]\n\n";

const char *usage =
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
"\n"
"The following options apply to only some routines.\n"
"  --nb x           Block size, default set automatically.\n"
"  --nrhs x         Number of right hand sides, default 1.\n"
"  --nstream x      Number of CUDA streams, default 1.\n"
"  --ngpu x         Number of GPUs, default 1. Also set with $MAGMA_NUM_GPUS.\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  --nthread x      Number of CPU threads, default 1.\n"
"  --itype [123]    Generalized Hermitian-definite eigenproblem type, default 1.\n"
"  --work  [123]    SVD workspace size, from min (1) to max (3), default 1.\n"
"  --version x      version of routine, e.g., during development, default 1.\n"
"  --fraction x     fraction of eigenvectors to compute, default 1.\n"
"  -L -U -F         uplo   = Lower*, Upper, or Full.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"  -U[NASO]         jobu   = No*, All, Some, or Overwrite; compute left  singular vectors.\n"
"  -V[NASO]         jobvt  = No*, All, Some, or Overwrite; compute right singular vectors.\n"
"  -J[NV]           jobz   = No* or Vectors; compute eigenvectors.\n"
"  -L[NV]           jobvl  = No* or Vectors; compute left  eigenvectors.\n"
"  -R[NV]           jobvr  = No* or Vectors; compute right eigenvectors.\n"
"                   * default values\n";

extern "C"
void parse_opts( int argc, char** argv, magma_opts *opts )
{
    // negative flag indicating -m, -n, -k not given
    int m = -1;
    int n = -1;
    int k = -1;
    
    // fill in default values
    opts->device   = 0;
    opts->nb       = 0;  // auto
    opts->nrhs     = 1;
    opts->nstream  = 1;
    opts->ngpu     = magma_num_gpus();
    opts->niter    = 1;
    opts->nthread  = 1;
    opts->itype    = 1;
    opts->svd_work = 1;
    opts->version  = 1;
    opts->fraction = 1.;
    
    opts->check     = (getenv("MAGMA_TESTINGS_CHECK") != NULL);
    opts->lapack    = (getenv("MAGMA_RUN_LAPACK")     != NULL);
    opts->warmup    = (getenv("MAGMA_WARMUP")         != NULL);
    opts->all       = (getenv("MAGMA_RUN_ALL")        != NULL);
    
    opts->uplo      = MagmaLower;      // potrf, etc.
    opts->transA    = MagmaNoTrans;    // gemm, etc.
    opts->transB    = MagmaNoTrans;    // gemm
    opts->side      = MagmaLeft;       // trsm, etc.
    opts->diag      = MagmaNonUnit;    // trsm, etc.
    opts->jobu      = MagmaNoVectors;  // gesvd: no left  singular vectors
    opts->jobvt     = MagmaNoVectors;  // gesvd: no right singular vectors
    opts->jobz      = MagmaNoVectors;  // heev:  no eigen vectors
    opts->jobvr     = MagmaNoVectors;  // geev:  no right eigen vectors
    opts->jobvl     = MagmaNoVectors;  // geev:  no left  eigen vectors
    
    printf( usage_short, argv[0] );
    
    int ndevices;
    cudaGetDeviceCount( &ndevices );
    
    int info;
    int ntest = 0;
    for( int i = 1; i < argc; ++i ) {
        // ----- matrix size
        // each -N fills in next entry of msize, nsize, ksize and increments ntest
        if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
            magma_assert( ntest < MAX_NTEST, "error: -N %s, max number of tests exceeded, ntest=%d.\n",
                          argv[i], ntest );
            i++;
            int m2, n2, k2;
            info = sscanf( argv[i], "%d,%d,%d", &m2, &n2, &k2 );
            if ( info == 3 && m2 >= 0 && n2 >= 0 && k2 >= 0 ) {
                opts->msize[ ntest ] = m2;
                opts->nsize[ ntest ] = n2;
                opts->ksize[ ntest ] = k2;
            }
            else if ( info == 2 && m2 >= 0 && n2 >= 0 ) {
                opts->msize[ ntest ] = m2;
                opts->nsize[ ntest ] = n2;
                opts->ksize[ ntest ] = n2;  // implicitly
            }
            else if ( info == 1 && m2 >= 0 ) {
                opts->msize[ ntest ] = m2;
                opts->nsize[ ntest ] = m2;  // implicitly
                opts->ksize[ ntest ] = m2;  // implicitly
            }
            else {
                fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, k >= 0.\n",
                         argv[i] );
                exit(1);
            }
            ntest++;
        }
        // --range start:stop:step fills in msize[ntest:], nsize[ntest:], ksize[ntest:]
        // with given range and updates ntest
        else if ( strcmp("--range", argv[i]) == 0 && i+1 < argc ) {
            i++;
            int start, stop, step;
            info = sscanf( argv[i], "%d:%d:%d", &start, &stop, &step );
            if ( info == 3 && start >= 0 && stop >= 0 && step != 0 ) {
                for( int n = start; (step > 0 ? n <= stop : n >= stop); n += step ) {
                    if ( ntest >= MAX_NTEST ) {
                        printf( "warning: --range %s, max number of tests reached, ntest=%d.\n",
                                argv[i], ntest );
                        break;
                    }
                    opts->msize[ ntest ] = n;
                    opts->nsize[ ntest ] = n;
                    opts->ksize[ ntest ] = n;
                    ntest++;
                }
            }
            else {
                fprintf( stderr, "error: --range %s is invalid; ensure start >= 0, stop >= start, step > 0.\n",
                         argv[i] );
                exit(1);
            }
        }
        // save m, n, k if -m, -n, -k is given; applied after loop
        else if ( strcmp("-m", argv[i]) == 0 && i+1 < argc ) {
            m = atoi( argv[++i] );
            magma_assert( m >= 0, "error: -m %s is invalid; ensure m >= 0.\n", argv[i] );
        }
        else if ( strcmp("-n", argv[i]) == 0 && i+1 < argc ) {
            n = atoi( argv[++i] );
            magma_assert( n >= 0, "error: -n %s is invalid; ensure n >= 0.\n", argv[i] );
        }
        else if ( strcmp("-k", argv[i]) == 0 && i+1 < argc ) {
            k = atoi( argv[++i] );
            magma_assert( k >= 0, "error: -k %s is invalid; ensure k >= 0.\n", argv[i] );
        }
        
        // ----- scalar arguments
        else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
            opts->device = atoi( argv[++i] );
            magma_assert( opts->device >= 0 && opts->device < ndevices,
                          "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
        }
        else if ( strcmp("--nrhs",    argv[i]) == 0 && i+1 < argc ) {
            opts->nrhs = atoi( argv[++i] );
            magma_assert( opts->nrhs >= 0,
                          "error: --nrhs %s is invalid; ensure nrhs >= 0.\n", argv[i] );
        }
        else if ( strcmp("--nb",      argv[i]) == 0 && i+1 < argc ) {
            opts->nb = atoi( argv[++i] );
            magma_assert( opts->nb > 0,
                          "error: --nb %s is invalid; ensure nb > 0.\n", argv[i] );
        }
        else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
            opts->ngpu = atoi( argv[++i] );
            magma_assert( opts->ngpu <= ndevices,
                          "error: --ngpu %s exceeds number of CUDA devices, %d.\n", argv[i], ndevices );
            magma_assert( opts->ngpu > 0,
                          "error: --ngpu %s is invalid; ensure ngpu > 0.\n", argv[i] );
        }
        else if ( strcmp("--nstream", argv[i]) == 0 && i+1 < argc ) {
            opts->nstream = atoi( argv[++i] );
            magma_assert( opts->nstream > 0,
                          "error: --nstream %s is invalid; ensure nstream > 0.\n", argv[i] );
        }
        else if ( strcmp("--niter",   argv[i]) == 0 && i+1 < argc ) {
            opts->niter = atoi( argv[++i] );
            magma_assert( opts->niter > 0,
                          "error: --niter %s is invalid; ensure niter > 0.\n", argv[i] );
        }
        else if ( strcmp("--nthread", argv[i]) == 0 && i+1 < argc ) {
            opts->nthread = atoi( argv[++i] );
            magma_assert( opts->nthread > 0,
                          "error: --nthread %s is invalid; ensure nthread > 0.\n", argv[i] );
        }
        else if ( strcmp("--itype",   argv[i]) == 0 && i+1 < argc ) {
            opts->itype = atoi( argv[++i] );
            magma_assert( opts->itype >= 1 && opts->itype <= 3,
                          "error: --itype %s is invalid; ensure itype in [1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--work",    argv[i]) == 0 && i+1 < argc ) {
            opts->svd_work = atoi( argv[++i] );
            magma_assert( opts->svd_work >= 1 && opts->svd_work <= 3,
                          "error: --work %s is invalid; ensure work in [1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--version", argv[i]) == 0 && i+1 < argc ) {
            opts->version = atoi( argv[++i] );
            magma_assert( opts->version >= 1,
                          "error: --version %s is invalid; ensure version > 0.\n", argv[i] );
        }
        else if ( strcmp("--fraction", argv[i]) == 0 && i+1 < argc ) {
            opts->fraction = atof( argv[++i] );
            magma_assert( opts->fraction >= 0 && opts->fraction <= 1,
                          "error: --fraction %s is invalid; ensure fraction in [0,1].\n", argv[i] );
        }
        
        // ----- boolean arguments
        // check results
        else if ( strcmp("-c",         argv[i]) == 0 ||
                  strcmp("--check",    argv[i]) == 0 ) { opts->check  = 1; }
        else if ( strcmp("-c2",        argv[i]) == 0 ||
                  strcmp("--check2",   argv[i]) == 0 ) { opts->check  = 2; }
        else if ( strcmp("--nocheck",  argv[i]) == 0 ) { opts->check  = 0; }
        else if ( strcmp("-l",         argv[i]) == 0 ||
                  strcmp("--lapack",   argv[i]) == 0 ) { opts->lapack = true;  }
        else if ( strcmp("--nolapack", argv[i]) == 0 ) { opts->lapack = false; }
        else if ( strcmp("--warmup",   argv[i]) == 0 ) { opts->warmup = true;  }
        else if ( strcmp("--nowarmup", argv[i]) == 0 ) { opts->warmup = false; }
        else if ( strcmp("--all",      argv[i]) == 0 ) { opts->all    = true;  }
        else if ( strcmp("--notall",   argv[i]) == 0 ) { opts->all    = false; }
        
        // ----- lapack flag arguments
        else if ( strcmp("-L",  argv[i]) == 0 ) { opts->uplo = MagmaLower; }
        else if ( strcmp("-U",  argv[i]) == 0 ) { opts->uplo = MagmaUpper; }
        else if ( strcmp("-F",  argv[i]) == 0 ) { opts->uplo = MagmaUpperLower; }
        
        else if ( strcmp("-NN", argv[i]) == 0 ) { opts->transA = MagmaNoTrans;   opts->transB = MagmaNoTrans;   }
        else if ( strcmp("-NT", argv[i]) == 0 ) { opts->transA = MagmaNoTrans;   opts->transB = MagmaTrans;     }
        else if ( strcmp("-NC", argv[i]) == 0 ) { opts->transA = MagmaNoTrans;   opts->transB = MagmaConjTrans; }
        else if ( strcmp("-TN", argv[i]) == 0 ) { opts->transA = MagmaTrans;     opts->transB = MagmaNoTrans;   }
        else if ( strcmp("-TT", argv[i]) == 0 ) { opts->transA = MagmaTrans;     opts->transB = MagmaTrans;     }
        else if ( strcmp("-TC", argv[i]) == 0 ) { opts->transA = MagmaTrans;     opts->transB = MagmaConjTrans; }
        else if ( strcmp("-CN", argv[i]) == 0 ) { opts->transA = MagmaConjTrans; opts->transB = MagmaNoTrans;   }
        else if ( strcmp("-CT", argv[i]) == 0 ) { opts->transA = MagmaConjTrans; opts->transB = MagmaTrans;     }
        else if ( strcmp("-CC", argv[i]) == 0 ) { opts->transA = MagmaConjTrans; opts->transB = MagmaConjTrans; }
        
        else if ( strcmp("-SL", argv[i]) == 0 ) { opts->side  = MagmaLeft;  }
        else if ( strcmp("-SR", argv[i]) == 0 ) { opts->side  = MagmaRight; }
        
        else if ( strcmp("-DN", argv[i]) == 0 ) { opts->diag  = MagmaNonUnit; }
        else if ( strcmp("-DU", argv[i]) == 0 ) { opts->diag  = MagmaUnit;    }
        
        else if ( strcmp("-UA", argv[i]) == 0 ) { opts->jobu  = MagmaAllVectors;       }
        else if ( strcmp("-US", argv[i]) == 0 ) { opts->jobu  = MagmaSomeVectors;      }
        else if ( strcmp("-UO", argv[i]) == 0 ) { opts->jobu  = MagmaOverwriteVectors; }
        else if ( strcmp("-UN", argv[i]) == 0 ) { opts->jobu  = MagmaNoVectors;        }
        
        else if ( strcmp("-VA", argv[i]) == 0 ) { opts->jobvt = MagmaAllVectors;       }
        else if ( strcmp("-VS", argv[i]) == 0 ) { opts->jobvt = MagmaSomeVectors;      }
        else if ( strcmp("-VO", argv[i]) == 0 ) { opts->jobvt = MagmaOverwriteVectors; }
        else if ( strcmp("-VN", argv[i]) == 0 ) { opts->jobvt = MagmaNoVectors;        }
        
        else if ( strcmp("-JN", argv[i]) == 0 ) { opts->jobz  = MagmaNoVectors; }
        else if ( strcmp("-JV", argv[i]) == 0 ) { opts->jobz  = MagmaVectors;   }
        
        else if ( strcmp("-LN", argv[i]) == 0 ) { opts->jobvl = MagmaNoVectors; }
        else if ( strcmp("-LV", argv[i]) == 0 ) { opts->jobvl = MagmaVectors;   }
        
        else if ( strcmp("-RN", argv[i]) == 0 ) { opts->jobvr = MagmaNoVectors; }
        else if ( strcmp("-RV", argv[i]) == 0 ) { opts->jobvr = MagmaVectors;   }
        
        // ----- usage
        else if ( strcmp("-h",     argv[i]) == 0 ||
                  strcmp("--help", argv[i]) == 0 ) {
            fprintf( stderr, usage, argv[0], MAX_NTEST );
            exit(0);
        }
        else {
            fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
            exit(1);
        }
    }
    
    // if -N or --range not given, use default range
    if ( ntest == 0 ) {
        int n2 = 1024 + 64;
        for( int i = 0; i < MAX_NTEST; ++i ) {
            opts->msize[i] = n2;
            opts->nsize[i] = n2;
            opts->ksize[i] = n2;
            n2 += 1024;
        }
        ntest = 10;
    }
    assert( ntest <= MAX_NTEST );
    opts->ntest = ntest;
    
    // fill in msize[:], nsize[:], ksize[:] if -m, -n, -k were given
    if ( m >= 0 ) {
        for( int j = 0; j < MAX_NTEST; ++j ) {
            opts->msize[j] = m;
        }
    }
    if ( n >= 0 ) {
        for( int j = 0; j < MAX_NTEST; ++j ) {
            opts->nsize[j] = n;
        }
    }
    if ( k >= 0 ) {
        for( int j = 0; j < MAX_NTEST; ++j ) {
            opts->ksize[j] = k;
        }
    }
    
    // find max dimensions
    opts->mmax = 0;
    opts->nmax = 0;
    opts->kmax = 0;
    for( int i = 0; i < ntest; ++i ) {
        opts->mmax = max( opts->mmax, opts->msize[i] );
        opts->nmax = max( opts->nmax, opts->nsize[i] );
        opts->kmax = max( opts->kmax, opts->ksize[i] );
    }
    
    // set device
    cudaSetDevice( opts->device );
}
// end parse_opts
