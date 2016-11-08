/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       
       Utilities for testing.
*/
#include <stdarg.h>
#include <string.h>
#include <assert.h>
#include <errno.h>

// flock exists only on Unix
#ifdef USE_FLOCK
#include <sys/file.h>  // flock
#include <sys/stat.h>  // fchmod
#endif

#include "magma_v2.h"
#include "testings.h"

// --------------------
// global variable
#if   defined(HAVE_CUBLAS)
    const char* g_platform_str = "cuBLAS";

#elif defined(HAVE_clBLAS)
    const char* g_platform_str = "clBLAS";

#elif defined(HAVE_MIC)
    const char* g_platform_str = "Xeon Phi";

#else
    #error "unknown platform"
#endif


// --------------------
// If condition is false, print error message and exit.
// Error message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        printf( "\n" );
        exit(1);
    }
}

// --------------------
// If condition is false, print warning message; does not exit.
// Warning message is formatted using printf, using any additional arguments.
extern "C"
void magma_assert_warn( bool condition, const char* msg, ... )
{
    if ( ! condition ) {
        printf( "Assert failed: " );
        va_list va;
        va_start( va, msg );
        vprintf( msg, va );
        printf( "\n" );
    }
}


// --------------------
// Acquire lock file.
// operation should be LOCK_SH (for shared access) or LOCK_EX (for exclusive access).
// Returns open file descriptor.
// Exits program on error.
// Lock is released by simply closing the file descriptor with close(),
// or when program exits or crashes.

int open_lockfile( const char* file, int operation )
{
    int fd = -1;
#ifdef USE_FLOCK
    int err;

    if ( file == NULL )
        return -1;
    else if ( operation != LOCK_SH && operation != LOCK_EX )
        return -2;
    
    fd = open( file, O_RDONLY|O_CREAT, 0666 );
    if ( fd < 0 ) {
        fprintf( stderr, "Error: Can't read file %s: %s (%d)\n",
                 file, strerror(errno), errno );
        exit(1);
    }

    // make it world-writable so anyone can rm the lockfile later on if needed
    // Ignore error -- occurs when someone else created the file.
    err = fchmod( fd, 0666 );
    //if ( err < 0 ) {
    //    fprintf( stderr, "Warning: Can't chmod file %s 0666: %s (%d)\n",
    //             file, strerror(errno), errno );
    //}
    
    // first try nonblocking lock;
    // if that fails (e.g., someone has exclusive lock) let user know and try blocking lock.
    err = flock( fd, operation|LOCK_NB );
    if ( err < 0 ) {
        fprintf( stderr, "Waiting for lock on %s...\n", file );
        err = flock( fd, operation );
        if ( err < 0 ) {
            fprintf( stderr, "Error: Can't lock file %s (operation %d): %s (%d)\n",
                     file, operation, strerror(errno), errno );
            exit(1);
        }
    }
#endif
    return fd;
}

// filename to use for lock file
const char* lockfile = "/tmp/icl-lock";


// --------------------
const char *usage_short =
"%% Usage: %s [options] [-h|--help]\n\n";

const char *usage =
"Options are:\n"
"  -n m[,n[,k]      Adds problem sizes. All of -n, -N, --range are now synonymous.\n"
"  -N m[,n[,k]      m, n, k can each be a single size or an inclusive range start:end:step.\n"
"  --range m[,n[,k] If two ranges are given, the number of sizes is limited by the smaller range.\n"
"                   If only m,n are given, then k=n. If only m is given, then n=k=m.\n"
"                   Examples:  -N 100  -N 100,200,300  -N 100,200:1000:100,300  -N 100:1000:100\n"
"  Default test sizes are the range 1088 : 10304 : 1024, that is, 1K+64 : 10K+64 : 1K.\n"
"  For batched, default sizes are     32 :   512 :   32.\n"
"\n"
"  -c  --[no]check  Whether to check results. Some tests always check.\n"
"                   Also set with $MAGMA_TESTINGS_CHECK.\n"
"  -c2 --check2     For getrf, check residual |Ax-b| instead of |PA-LU|.\n"
"  -l  --[no]lapack Whether to run lapack. Some tests always run lapack.\n"
"                   Also set with $MAGMA_RUN_LAPACK.\n"
"      --[no]warmup Whether to warmup. Not yet implemented in most cases.\n"
"                   Also set with $MAGMA_WARMUP.\n"
"  --dev x          GPU device to use, default 0.\n"
"  --align n        Round up LDDA on GPU to multiple of align, default 32.\n"
"  --verbose        Verbose output.\n"
"  -x  --exclusive  Lock file for exclusive use (internal ICL functionality).\n"
"\n"
"The following options apply to only some routines.\n"
"  --batch x        number of matrices for the batched routines, default 1000.\n"
"  --nb x           Block size, default set automatically.\n"
"  --nrhs x         Number of right hand sides, default 1.\n"
"  --nqueue x       Number of device queues, default 1.\n"
"  --ngpu x         Number of GPUs, default 1. Also set with $MAGMA_NUM_GPUS.\n"
"                   (Some testers take --ngpu -1 to run the multi-GPU code with 1 GPU.\n"
"  --nsub x         Number of submatrices, default 1.\n"
"  --niter x        Number of iterations to repeat each test, default 1.\n"
"  --nthread x      Number of CPU threads for some experimental codes, default 1.\n"
"                   (For most testers, set $OMP_NUM_THREADS or $MKL_NUM_THREADS\n"
"                    to control the number of CPU threads.)\n"
"  --offset x       Offset from beginning of matrix, default 0.\n"
"  --itype [123]    Generalized Hermitian-definite eigenproblem type, default 1.\n"
"  --svd-work x     SVD workspace size, one of:\n"
"         query*    queries LAPACK and MAGMA\n"
"         doc       is what LAPACK and MAGMA document as required\n"
"         doc_old   is what LAPACK <= 3.6 documents\n"
"         min       is minimum required, which may be smaller than doc\n"
"         min_old   is minimum required by LAPACK <= 3.6\n"
"         min_fast  is minimum to take fast path in gesvd\n"
"         min-1     is (minimum - 1), to test error return\n"
"         opt       is optimal\n"
"         opt_old   is optimal as computed by LAPACK <= 3.6\n"
"         opt_slow  is optimal for slow path in gesvd\n"
"         max       is maximum that will be used\n"
"\n"
"  --version x      version of routine, e.g., during development, default 1.\n"
"  --fraction x     fraction of eigenvectors to compute, default 1.\n"
"                   If fraction == 0, computes eigenvalues il=0.1*N to iu=0.3*N.\n"
"  --tolerance x    accuracy tolerance, multiplied by machine epsilon, default 30.\n"
"  --tol x          same.\n"
"  -L -U -F         uplo   = Lower*, Upper, or Full.\n"
"  -[NTC][NTC]      transA = NoTrans*, Trans, or ConjTrans (first letter) and\n"
"                   transB = NoTrans*, Trans, or ConjTrans (second letter).\n"
"  -[TC]            transA = Trans or ConjTrans. Default is NoTrans. Doesn't change transB.\n"
"  -S[LR]           side   = Left*, Right.\n"
"  -D[NU]           diag   = NonUnit*, Unit.\n"
"  --jobu [nsoa]    No*, Some, Overwrite, or All left singular vectors (U). gesdd uses this for jobz.\n"
"  --jobv [nsoa]    No*, Some, Overwrite, or All right singular vectors (V).\n"
"  -J[NV]           jobz   = No* or Vectors; compute eigenvectors (symmetric).\n"
"  -L[NV]           jobvl  = No* or Vectors; compute left  eigenvectors (non-symmetric).\n"
"  -R[NV]           jobvr  = No* or Vectors; compute right eigenvectors (non-symmetric).\n"
"\n"
"                   * default values\n";


// constructor fills in default values
magma_opts::magma_opts( magma_opts_t flag )
{
    // fill in default values
    this->batchcount = 300;
    this->device   = 0;
    this->align    = 32;
    this->nb       = 0;  // auto
    this->nrhs     = 1;
    this->nqueue   = 1;
    this->ngpu     = magma_num_gpus();
    this->nsub     = 1;
    this->niter    = 1;
    this->nthread  = 1;
    this->offset   = 0;
    this->itype    = 1;
    this->version  = 1;
    this->verbose  = 0;
    this->fraction = 1.;
    this->tolerance = 30.;
    this->check     = (getenv("MAGMA_TESTINGS_CHECK") != NULL);
    this->magma     = true;
    this->lapack    = (getenv("MAGMA_RUN_LAPACK")     != NULL);
    this->warmup    = (getenv("MAGMA_WARMUP")         != NULL);
    
    this->uplo      = MagmaLower;      // potrf, etc.
    this->transA    = MagmaNoTrans;    // gemm, etc.
    this->transB    = MagmaNoTrans;    // gemm
    this->side      = MagmaLeft;       // trsm, etc.
    this->diag      = MagmaNonUnit;    // trsm, etc.
    this->jobz      = MagmaNoVec;  // heev:  no eigen vectors
    this->jobvr     = MagmaNoVec;  // geev:  no right eigen vectors
    this->jobvl     = MagmaNoVec;  // geev:  no left  eigen vectors
    
    #ifdef USE_FLOCK
    this->flock_op = LOCK_SH;  // default shared lock
    #endif
    
    if ( flag == MagmaOptsBatched ) {
        // 32, 64, ..., 512
        this->default_nstart = 32;
        this->default_nstep  = 32;
        this->default_nend   = 512;
    }
    else {
        // 1K + 64, 2K + 64, ..., 10K + 64
        this->default_nstart = 1024 + 64;
        this->default_nstep  = 1024;
        this->default_nend   = 10304;
    }
}


// Given pointer to a string, scans the string for a comma,
// and advances the string to after the comma.
// Returns true if comma found, otherwise false.
bool scan_comma( char** handle )
{
    char* ptr = *handle;
    // scan past whitespace
    while( *ptr == ' ' ) {
        ptr += 1;
    }
    // scan comma
    if ( *ptr == ',' ) {
        *handle = ptr + 1;
        return true;
    }
    else {
        return false;
    }
}


// Given pointer to a string, scans the string for a range "%d:%d:%d" or a number "%d".
// If range,  then start, end, step are set accordingly.
// If number, then start = end and step = 0.
// Advances the string to after the range or number.
// Ensures start, end >= 0.
// If step >= 0, ensures start <= end;
// if step <  0, ensures start >= end.
// Returns true if found valid range or number, otherwise false.
bool scan_range( char** handle, int* start, int* end, int* step )
{
    int bytes1, bytes3, cnt;
    char* ptr = *handle;
    cnt = sscanf( ptr, "%d%n:%d:%d%n", start, &bytes1, end, step, &bytes3 );
    if ( cnt == 3 ) {
        *handle += bytes3;
        return (*start >= 0 && *end >= 0 && (*step >= 0 ? *start <= *end : *start >= *end));
    }
    else if ( cnt == 1 ) {
        *handle += bytes1;
        *end  = *start;
        *step = 0;
        return (*start >= 0);
    }
    else {
        return false;
    }
}


// parse values from command line
void magma_opts::parse_opts( int argc, char** argv )
{
    printf( usage_short, argv[0] );
    
    magma_int_t ndevices;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices( devices, MagmaMaxGPUs, &ndevices );
    
    this->ntest = 0;
    for( int i = 1; i < argc; ++i ) {
        // ----- problem size
        // -n or -N or --range fill in single size or range of sizes, and update ntest
        if ( (strcmp("-n",      argv[i]) == 0 ||
              strcmp("-N",      argv[i]) == 0 ||
              strcmp("--range", argv[i]) == 0) && i+1 < argc )
        {
            i++;
            int m_start, m_end, m_step;
            int n_start, n_end, n_step;
            int k_start, k_end, k_step;
            char* ptr = argv[i];
            bool valid = scan_range( &ptr, &m_start, &m_end, &m_step );
            if ( valid ) {
                if ( *ptr == '\0' ) {
                    n_start = k_start = m_start;
                    n_end   = k_end   = m_end;
                    n_step  = k_step  = m_step;
                }
                else {
                    valid = scan_comma( &ptr ) && scan_range( &ptr, &n_start, &n_end, &n_step );
                    if ( valid ) {
                        if ( *ptr == '\0' ) {
                            k_start = n_start;
                            k_end   = n_end;
                            k_step  = n_step;
                        }
                        else {
                            valid = scan_comma( &ptr ) && scan_range( &ptr, &k_start, &k_end, &k_step );
                            valid = (valid && *ptr == '\0');
                        }
                    }
                }
            }
            
            magma_assert( valid, "error: '%s %s' is not valid, expected (m|m_start:m_end:m_step)[,(n|n_start:n_end:n_step)[,(k|k_start:k_end:k_step)]]\n",
                          argv[i-1], argv[i] );
            // if all zero steps, just give start point
            if ( m_step == 0 && n_step == 0 && k_step == 0 ) {
                magma_assert( this->ntest < MAX_NTEST, "error: %s %s exceeded maximum number of tests (%d).\n",
                              argv[i-1], argv[i], MAX_NTEST );
                this->msize[ this->ntest ] = m_start;
                this->nsize[ this->ntest ] = n_start;
                this->ksize[ this->ntest ] = k_start;
                this->ntest++;
            }
            else {
                for( int m=m_start, n=n_start, k=k_start;
                     (m_step >= 0 ? m <= m_end : m >= m_end) &&
                     (n_step >= 0 ? n <= n_end : n >= n_end) &&
                     (k_step >= 0 ? k <= k_end : k >= k_end);
                     m += m_step, n += n_step, k += k_step )
                {
                    magma_assert( this->ntest < MAX_NTEST, "error: %s %s exceeded maximum number of tests (%d).\n",
                                  argv[i-1], argv[i], MAX_NTEST );
                    this->msize[ this->ntest ] = m;
                    this->nsize[ this->ntest ] = n;
                    this->ksize[ this->ntest ] = k;
                    this->ntest++;
                }
            }
        }
        
        // ----- scalar arguments
        else if ( strcmp("--dev", argv[i]) == 0 && i+1 < argc ) {
            this->device = atoi( argv[++i] );
            magma_assert( this->device >= 0 && this->device < ndevices,
                          "error: --dev %s is invalid; ensure dev in [0,%d].\n", argv[i], ndevices-1 );
        }
        else if ( strcmp("--align", argv[i]) == 0 && i+1 < argc ) {
            this->align = atoi( argv[++i] );
            magma_assert( this->align >= 1 && this->align <= 4096,
                          "error: --align %s is invalid; ensure align in [1,4096].\n", argv[i] );
        }
        else if ( strcmp("--nrhs",    argv[i]) == 0 && i+1 < argc ) {
            this->nrhs = atoi( argv[++i] );
            magma_assert( this->nrhs >= 0,
                          "error: --nrhs %s is invalid; ensure nrhs >= 0.\n", argv[i] );
        }
        else if ( strcmp("--nb",      argv[i]) == 0 && i+1 < argc ) {
            this->nb = atoi( argv[++i] );
            magma_assert( this->nb > 0,
                          "error: --nb %s is invalid; ensure nb > 0.\n", argv[i] );
        }
        else if ( strcmp("--ngpu",    argv[i]) == 0 && i+1 < argc ) {
            this->ngpu = atoi( argv[++i] );
            magma_assert( this->ngpu <= MagmaMaxGPUs,
                          "error: --ngpu %s exceeds MagmaMaxGPUs, %d.\n", argv[i], MagmaMaxGPUs );
            magma_assert( this->ngpu <= ndevices,
                          "error: --ngpu %s exceeds number of CUDA or OpenCL devices, %d.\n", argv[i], ndevices );
            // allow ngpu == -1, which forces multi-GPU code with 1 GPU. see testing_zhegvd, etc.
            magma_assert( this->ngpu > 0 || this->ngpu == -1,
                          "error: --ngpu %s is invalid; ensure ngpu != 0.\n", argv[i] );
            // save in environment variable, so magma_num_gpus() picks it up
            char env_num_gpus[20];  // space for "MAGMA_NUM_GPUS=", 4 digits, and nil
            #if defined( _WIN32 ) || defined( _WIN64 )
                snprintf( env_num_gpus, sizeof(env_num_gpus), "MAGMA_NUM_GPUS=%lld", (long long) abs(this->ngpu) );
                putenv( env_num_gpus );
            #else
                snprintf( env_num_gpus, sizeof(env_num_gpus), "%lld", (long long) abs(this->ngpu) );
                setenv( "MAGMA_NUM_GPUS", env_num_gpus, true );
            #endif
        }
        else if ( strcmp("--nsub", argv[i]) == 0 && i+1 < argc ) {
            this->nsub = atoi( argv[++i] );
            magma_assert( this->nsub > 0,
                          "error: --nsub %s is invalid; ensure nsub > 0.\n", argv[i] );
        }
        else if ( strcmp("--nqueue", argv[i]) == 0 && i+1 < argc ) {
            this->nqueue = atoi( argv[++i] );
            magma_assert( this->nqueue > 0,
                          "error: --nqueue %s is invalid; ensure nqueue > 0.\n", argv[i] );
        }
        else if ( strcmp("--niter",   argv[i]) == 0 && i+1 < argc ) {
            this->niter = atoi( argv[++i] );
            magma_assert( this->niter > 0,
                          "error: --niter %s is invalid; ensure niter > 0.\n", argv[i] );
        }
        else if ( strcmp("--nthread", argv[i]) == 0 && i+1 < argc ) {
            this->nthread = atoi( argv[++i] );
            magma_assert( this->nthread > 0,
                          "error: --nthread %s is invalid; ensure nthread > 0.\n", argv[i] );
        }
        else if ( strcmp("--offset", argv[i]) == 0 && i+1 < argc ) {
            this->offset = atoi( argv[++i] );
            magma_assert( this->offset >= 0,
                          "error: --offset %s is invalid; ensure offset >= 0.\n", argv[i] );
        }
        else if ( strcmp("--itype",   argv[i]) == 0 && i+1 < argc ) {
            this->itype = atoi( argv[++i] );
            magma_assert( this->itype >= 1 && this->itype <= 3,
                          "error: --itype %s is invalid; ensure itype in [1,2,3].\n", argv[i] );
        }
        else if ( strcmp("--version", argv[i]) == 0 && i+1 < argc ) {
            this->version = atoi( argv[++i] );
            magma_assert( this->version >= 1,
                          "error: --version %s is invalid; ensure version > 0.\n", argv[i] );
        }
        else if ( strcmp("--fraction", argv[i]) == 0 && i+1 < argc ) {
            this->fraction = atof( argv[++i] );
            magma_assert( this->fraction >= 0 && this->fraction <= 1,
                          "error: --fraction %s is invalid; ensure fraction in [0,1].\n", argv[i] );
        }
        else if ( (strcmp("--tol",       argv[i]) == 0 ||
                   strcmp("--tolerance", argv[i]) == 0) && i+1 < argc ) {
            this->tolerance = atof( argv[++i] );
            magma_assert( this->tolerance >= 0 && this->tolerance <= 1000,
                          "error: --tolerance %s is invalid; ensure tolerance in [0,1000].\n", argv[i] );
        }
        else if ( strcmp("--batch", argv[i]) == 0 && i+1 < argc ) {
            this->batchcount = atoi( argv[++i] );
            magma_assert( this->batchcount > 0,
                          "error: --batch %s is invalid; ensure batch > 0.\n", argv[i] );
        }
        // ----- boolean arguments
        // check results
        else if ( strcmp("-c",         argv[i]) == 0 ||
                  strcmp("--check",    argv[i]) == 0 ) { this->check  = 1; }
        else if ( strcmp("-c2",        argv[i]) == 0 ||
                  strcmp("--check2",   argv[i]) == 0 ) { this->check  = 2; }
        else if ( strcmp("--nocheck",  argv[i]) == 0 ) { this->check  = 0; }
        
        else if ( strcmp("-l",         argv[i]) == 0 ||
                  strcmp("--lapack",   argv[i]) == 0 ) { this->lapack = true;  }
        else if ( strcmp("--nolapack", argv[i]) == 0 ) { this->lapack = false; }
        
        else if ( strcmp("--magma",    argv[i]) == 0 ) { this->magma  = true;  }
        else if ( strcmp("--nomagma",  argv[i]) == 0 ) { this->magma  = false; }
        
        else if ( strcmp("--warmup",   argv[i]) == 0 ) { this->warmup = true;  }
        else if ( strcmp("--nowarmup", argv[i]) == 0 ) { this->warmup = false; }
        
        //else if ( strcmp("--all",      argv[i]) == 0 ) { this->all    = true;  }
        //else if ( strcmp("--notall",   argv[i]) == 0 ) { this->all    = false; }
        
        else if ( strcmp("-v",         argv[i]) == 0 ||
                  strcmp("--verbose",  argv[i]) == 0 ) { this->verbose += 1;  }
        
        // ----- lapack options
        else if ( strcmp("-L",  argv[i]) == 0 ) { this->uplo = MagmaLower; }
        else if ( strcmp("-U",  argv[i]) == 0 ) { this->uplo = MagmaUpper; }
        else if ( strcmp("-F",  argv[i]) == 0 ) { this->uplo = MagmaFull; }
        
        else if ( strcmp("-NN", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaNoTrans;   }
        else if ( strcmp("-NT", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaTrans;     }
        else if ( strcmp("-NC", argv[i]) == 0 ) { this->transA = MagmaNoTrans;   this->transB = MagmaConjTrans; }
        else if ( strcmp("-TN", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaNoTrans;   }
        else if ( strcmp("-TT", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaTrans;     }
        else if ( strcmp("-TC", argv[i]) == 0 ) { this->transA = MagmaTrans;     this->transB = MagmaConjTrans; }
        else if ( strcmp("-CN", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaNoTrans;   }
        else if ( strcmp("-CT", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaTrans;     }
        else if ( strcmp("-CC", argv[i]) == 0 ) { this->transA = MagmaConjTrans; this->transB = MagmaConjTrans; }
        else if ( strcmp("-T",  argv[i]) == 0 ) { this->transA = MagmaTrans;     }
        else if ( strcmp("-C",  argv[i]) == 0 ) { this->transA = MagmaConjTrans; }
        
        else if ( strcmp("-SL", argv[i]) == 0 ) { this->side  = MagmaLeft;  }
        else if ( strcmp("-SR", argv[i]) == 0 ) { this->side  = MagmaRight; }
        
        else if ( strcmp("-DN", argv[i]) == 0 ) { this->diag  = MagmaNonUnit; }
        else if ( strcmp("-DU", argv[i]) == 0 ) { this->diag  = MagmaUnit;    }
        
        else if ( strcmp("-JN", argv[i]) == 0 ) { this->jobz  = MagmaNoVec; }
        else if ( strcmp("-JV", argv[i]) == 0 ) { this->jobz  = MagmaVec;   }
        
        else if ( strcmp("-LN", argv[i]) == 0 ) { this->jobvl = MagmaNoVec; }
        else if ( strcmp("-LV", argv[i]) == 0 ) { this->jobvl = MagmaVec;   }
        
        else if ( strcmp("-RN", argv[i]) == 0 ) { this->jobvr = MagmaNoVec; }
        else if ( strcmp("-RV", argv[i]) == 0 ) { this->jobvr = MagmaVec;   }
        
        // ----- vectors of options
        else if ( strcmp("--svd-work", argv[i]) == 0 && i+1 < argc ) {
            i += 1;
            char *token;
            char *arg = strdup( argv[i] );
            for (token = strtok( arg, ", " );
                 token != NULL;
                 token = strtok( NULL, ", " ))
            {
                if ( *token == '\0' ) { /* ignore empty tokens */ }
                else if ( strcmp( token, "all"       ) == 0 ) { this->svd_work.push_back( MagmaSVD_all        ); }
                else if ( strcmp( token, "query"     ) == 0 ) { this->svd_work.push_back( MagmaSVD_query      ); }
                else if ( strcmp( token, "doc"       ) == 0 ) { this->svd_work.push_back( MagmaSVD_doc        ); }
                else if ( strcmp( token, "doc_old"   ) == 0 ) { this->svd_work.push_back( MagmaSVD_doc_old    ); }
                else if ( strcmp( token, "min"       ) == 0 ) { this->svd_work.push_back( MagmaSVD_min        ); }
                else if ( strcmp( token, "min-1"     ) == 0 ) { this->svd_work.push_back( MagmaSVD_min_1      ); }
                else if ( strcmp( token, "min_old"   ) == 0 ) { this->svd_work.push_back( MagmaSVD_min_old    ); }
                else if ( strcmp( token, "min_old-1" ) == 0 ) { this->svd_work.push_back( MagmaSVD_min_old_1  ); }
                else if ( strcmp( token, "min_fast"  ) == 0 ) { this->svd_work.push_back( MagmaSVD_min_fast   ); }
                else if ( strcmp( token, "min_fast-1") == 0 ) { this->svd_work.push_back( MagmaSVD_min_fast_1 ); }
                else if ( strcmp( token, "opt"       ) == 0 ) { this->svd_work.push_back( MagmaSVD_opt        ); }
                else if ( strcmp( token, "opt_old"   ) == 0 ) { this->svd_work.push_back( MagmaSVD_opt_old    ); }
                else if ( strcmp( token, "opt_slow"  ) == 0 ) { this->svd_work.push_back( MagmaSVD_opt_slow   ); }
                else if ( strcmp( token, "max"       ) == 0 ) { this->svd_work.push_back( MagmaSVD_max        ); }
                else {
                    magma_assert( false, "error: --svd-work '%s' is invalid\n", argv[i] );
                }
            }
            free( arg );
        }
        
        else if ( strcmp("--jobu", argv[i]) == 0 && i+1 < argc ) {
            i += 1;
            const char* arg = argv[i];
            while( *arg != '\0' ) {
                this->jobu.push_back( magma_vec_const( *arg ));
                ++arg;
                if ( *arg == ',' )
                    ++arg;
            }
        }
        else if ( (strcmp("--jobv",  argv[i]) == 0 ||
                   strcmp("--jobvt", argv[i]) == 0) && i+1 < argc ) {
            i += 1;
            const char* arg = argv[i];
            while( *arg != '\0' ) {
                this->jobv.push_back( magma_vec_const( *arg ));
                ++arg;
                if ( *arg == ',' )
                    ++arg;
            }
        }
        
        // ----- misc
        else if ( strcmp("-x",          argv[i]) == 0 ||
                  strcmp("--exclusive", argv[i]) == 0 ) {
            #ifdef USE_FLOCK
            this->flock_op = LOCK_EX;
            #else
            fprintf( stderr, "ignoring %s: USE_FLOCK not defined; flock not supported.\n", argv[i] );
            #endif
        }
        
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
    
    // default values
    if ( this->svd_work.size() == 0 ) {
        this->svd_work.push_back( MagmaSVD_query );
    }
    if ( this->jobu.size() == 0 ) {
        this->jobu.push_back( MagmaNoVec );
    }
    if ( this->jobv.size() == 0 ) {
        this->jobv.push_back( MagmaNoVec );
    }
    
    // if -N or --range not given, use default range
    if ( this->ntest == 0 ) {
        magma_int_t n2 = this->default_nstart;  //1024 + 64;
        while( n2 <= this->default_nend && this->ntest < MAX_NTEST ) {
            this->msize[ this->ntest ] = n2;
            this->nsize[ this->ntest ] = n2;
            this->ksize[ this->ntest ] = n2;
            n2 += this->default_nstep;  //1024;
            this->ntest++;
        }
    }
    assert( this->ntest <= MAX_NTEST );
    
    // lock file
    #ifdef USE_FLOCK
    this->flock_fd = open_lockfile( lockfile, this->flock_op );
    #endif

    #ifdef HAVE_CUBLAS
    magma_setdevice( this->device );
    #endif
    
    // create queues on this device
    // 2 queues + 1 extra NULL entry to catch errors
    magma_queue_create( devices[ this->device ], &this->queues2[ 0 ] );
    magma_queue_create( devices[ this->device ], &this->queues2[ 1 ] );
    this->queues2[ 2 ] = NULL;
    
    this->queue = this->queues2[ 0 ];
    
    #ifdef HAVE_CUBLAS
    // handle for directly calling cublas
    this->handle = magma_queue_get_cublas_handle( this->queue );
    #endif
}
// end parse_opts


// ------------------------------------------------------------
void magma_opts::cleanup()
{
    this->queue = NULL;
    magma_queue_destroy( this->queues2[0] );
    magma_queue_destroy( this->queues2[1] );
    this->queues2[0] = NULL;
    this->queues2[1] = NULL;
    
    #ifdef HAVE_CUBLAS
    this->handle = NULL;
    #endif
}


// ------------------------------------------------------------
// Initialize PAPI events set to measure flops.
// Note flops counters are inaccurate on Sandy Bridge, and don't exist on Haswell.
// See http://icl.cs.utk.edu/projects/papi/wiki/PAPITopics:SandyFlops
#ifdef HAVE_PAPI
#include <papi.h>
#include <string.h>  // memset
#endif  // HAVE_PAPI

int gPAPI_flops_set = -1;  // i.e., PAPI_NULL

extern "C"
void flops_init()
{
    #ifdef HAVE_PAPI
    int err = PAPI_library_init( PAPI_VER_CURRENT );
    if ( err != PAPI_VER_CURRENT ) {
        fprintf( stderr, "Error: PAPI couldn't initialize: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    // read flops
    err = PAPI_create_eventset( &gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_create_eventset failed\n" );
    }
    
    err = PAPI_assign_eventset_component( gPAPI_flops_set, 0 );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_assign_eventset_component failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    PAPI_option_t opt;
    memset( &opt, 0, sizeof(PAPI_option_t) );
    opt.inherit.inherit  = PAPI_INHERIT_ALL;
    opt.inherit.eventset = gPAPI_flops_set;
    err = PAPI_set_opt( PAPI_INHERIT, &opt );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_set_opt failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    err = PAPI_add_event( gPAPI_flops_set, PAPI_FP_OPS );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_add_event failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    
    err = PAPI_start( gPAPI_flops_set );
    if ( err != PAPI_OK ) {
        fprintf( stderr, "Error: PAPI_start failed: %s (%d)\n",
                 PAPI_strerror(err), err );
    }
    #endif  // HAVE_PAPI
}
