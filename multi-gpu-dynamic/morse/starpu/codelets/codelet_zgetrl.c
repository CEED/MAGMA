/**
 *
 *  @file codelet_zgetrf_incpiv.c
 *
 *  MAGMA codelets kernel
 *  MAGMA is a software package provided by Univ. of Tennessee,
 *  Univ. of California Berkeley and Univ. of Colorado Denver,
 *  and INRIA Bordeaux Sud-Ouest
 *
 *  @version 2.3.1
 *  @author Mathieu Faverge
 *  @author Cedric Augonnet
 *  @date 2011-06-01
 *  @precisions normal z -> c d s
 *
 **/
#include "morse_starpu.h"

#if (PLASMA_VERSION_MAJOR >= 2) && (PLASMA_VERSION_MINOR >= 4)
#define CORE_zgetrf CORE_zgetrf_incpiv
#endif

/*
 * Codelet CPU
 */
static void cl_zgetrf_incpiv_cpu_func(void *descr[], void *cl_arg)
{
    int m;
    int n;
    int ib;
    PLASMA_Complex64_t *A;
    int lda;
    int ldl;
    int *IPIV;
    morse_starpu_ws_t *h_work;
    morse_starpu_ws_t *d_work;
    PLASMA_bool check_info;
    int iinfo;
    int info;

    A = (PLASMA_Complex64_t *)STARPU_MATRIX_GET_PTR(descr[0]);

    starpu_codelet_unpack_args(cl_arg, &m, &n, &ib, &lda, &ldl, &IPIV, &h_work, &d_work, &check_info, &iinfo);
    CORE_zgetrf(m, n, ib, A, lda, IPIV, &info);

#if defined(MORSE_USE_CUDA) && !defined(WITHOUTTRTRI)
    PLASMA_Complex64_t *L;
    L = (PLASMA_Complex64_t *)STARPU_MATRIX_GET_PTR(descr[1]);

    /*
     * L stores:
     *      L1     L2    L3     ...
     *      L1^-1  L2^-1 L3^-1  ...
     */
    /* Compute L-1 in lower rectangle of L */
    L += ib;
    { 
        int i, sb;
        for (i=0; i<n; i+=ib) {
            sb = min( ib, n-i );
            CORE_zlacpy(PlasmaUpperLower, sb, sb, A+(i*lda+i), lda, L+(i*ldl), ldl );
            
            CORE_ztrtri( PlasmaLower, PlasmaUnit, sb, L+(i*ldl), ldl, &info );
            if (info != 0 ) {
                fprintf(stderr, "ERROR, trtri returned with info = %d\n", info);
            }          
        }
    }
#endif

    /* if (check_info && info != PLASMA_SUCCESS) */
    /*     return iinfo+info */
}

/*
 * Codelet Multi-cores
 */
#ifdef MORSE_USE_MULTICORE
static void cl_zgetrf_incpiv_mc_func(void *descr[], void *cl_arg)
{
}
#endif

/*
 * Codelet GPU
 */
#ifdef MORSE_USE_CUDA
static void cl_zgetrf_incpiv_cuda_func(void *descr[], void *cl_arg)
{
    int m;
    int n;
    int ib;
    cuDoubleComplex *hA, *dA;
    cuDoubleComplex *hL, *dL;
    cuDoubleComplex *dwork;
    morse_starpu_ws_t *h_work;
    morse_starpu_ws_t *d_work;
    int lda, ldl;
    int *IPIV;
    PLASMA_bool check_info;
    int iinfo;
    int info;

    starpu_codelet_unpack_args(cl_arg, &m, &n, &ib, &lda, &ldl, &IPIV, &h_work, &d_work, &check_info, &iinfo);

    dA = (cuDoubleComplex *)STARPU_MATRIX_GET_PTR(descr[0]);
    dL = (cuDoubleComplex *)STARPU_MATRIX_GET_PTR(descr[1]);
    /*
     * hwork => at least (2*IB+NB)*NB contains all hA and hL
     * dwork => at least IB*NB
     */
    hA    = morse_starpu_ws_getlocal(h_work);
    dwork = morse_starpu_ws_getlocal(d_work);

    hL = hA + lda*n;
    
    /* Initialize L to 0 */
    memset(hL, 0, ldl*n*sizeof(cuDoubleComplex));

    /* Copy First panel */
    cublasGetMatrix( m, min(ib,m), sizeof(cuDoubleComplex), dA, lda, hA, lda );

    magma_zgetrf_incpiv_gpu( 'C', m, n, ib,
                      hA, lda, dA, lda,
                      hL, ldl, dL, ldl,
                      IPIV, 
                      dwork, lda,
                      &info );

    cudaThreadSynchronize();
}
#endif

/*
 * Codelet definition
 */
CODELETS(zgetrf_incpiv, 2, cl_zgetrf_incpiv_cpu_func, cl_zgetrf_incpiv_cuda_func, cl_zgetrf_incpiv_cpu_func)

/*
 * Wrapper
 */
void MORSE_zgetrf_incpiv( MorseOption_t *options, 
                   int m, int n, int ib,
                   magma_desc_t *A, int Am, int An,
                   magma_desc_t *L, int Lm, int Ln,
                   int *IPIV,
                   PLASMA_bool check, int iinfo)
{
    struct starpu_codelet *zgetrf_incpiv_codelet;
    void (*callback)(void*) = options->profiling ? cl_zgetrf_incpiv_callback : NULL;
    int lda = BLKLDD( A, Am );
    int ldl = BLKLDD( L, Lm );
    morse_starpu_ws_t *h_work = (morse_starpu_ws_t*)(options->ws_host);
    morse_starpu_ws_t *d_work = (morse_starpu_ws_t*)(options->ws_device);

#ifdef MORSE_USE_MULTICORE
    zgetrf_incpiv_codelet = options->parallel ? &cl_zgetrf_incpiv_mc : &cl_zgetrf_incpiv;
#else
    zgetrf_incpiv_codelet = &cl_zgetrf_incpiv;
#endif

    starpu_insert_task(zgetrf_incpiv_codelet,
                       STARPU_VALUE,  &m,      sizeof(int),
                       STARPU_VALUE,  &n,      sizeof(int),
                       STARPU_VALUE,  &ib,     sizeof(int),
                       STARPU_RW,  BLKADDR( A, PLASMA_Complex64_t, Am, An ),
                       STARPU_VALUE,  &lda,    sizeof(int),
                       STARPU_W, BLKADDR( L, PLASMA_Complex64_t, Lm, Ln ),
                       STARPU_VALUE,  &ldl,    sizeof(int),
                       STARPU_VALUE,  &IPIV,   sizeof(int*),
                       STARPU_VALUE,  &h_work, sizeof(morse_starpu_ws_t*),
                       STARPU_VALUE,  &d_work, sizeof(morse_starpu_ws_t*),
                       STARPU_VALUE,  &check,  sizeof(PLASMA_bool),
                       STARPU_VALUE,  &iinfo,  sizeof(int),
                       STARPU_PRIORITY,     options->priority,
                       STARPU_CALLBACK,     callback, NULL,
                       0);
}
