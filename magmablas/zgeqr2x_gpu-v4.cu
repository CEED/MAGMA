/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 512
#else
   #define BLOCK_SIZE 768
#endif

extern "C" void
magma_zlarfbx_gpu(int m, int k, cuDoubleComplex *V, int ldv,
                  cuDoubleComplex *dT, int ldt, cuDoubleComplex *c,
                  cuDoubleComplex *dwork);

extern "C" void
magma_zlarfgtx_gpu(int n, cuDoubleComplex *dx0, cuDoubleComplex *dx,
                   cuDoubleComplex *dtau, double *dxnorm,
                   cuDoubleComplex *dA, int it,
                   cuDoubleComplex *V, int ldv, cuDoubleComplex *T, int ldt,
                   cuDoubleComplex *dwork);

extern "C" void
magmablas_dznrm2_adjust(int k, double *xnorm, cuDoubleComplex *c);
    
extern "C" void
magmablas_zgemm_reduce(magma_int_t m, magma_int_t n, magma_int_t k,
                       cuDoubleComplex alpha, const cuDoubleComplex *d_A, magma_int_t lda,
                       const cuDoubleComplex *d_B, magma_int_t ldb,
                       cuDoubleComplex beta,        cuDoubleComplex *d_C, magma_int_t ldc );

__global__ void 
magma_ztrmv_kernel2(const cuDoubleComplex *T, int ldt,
                    cuDoubleComplex *v, cuDoubleComplex *y, cuDoubleComplex *tau);

__global__ void 
magma_zgemv_kernel3(int m, const cuDoubleComplex * __restrict__ V, int ldv,
                    cuDoubleComplex *c, cuDoubleComplex *dwork,
                    cuDoubleComplex *tau);


extern "C" magma_int_t
magma_zlarfb2_gpu( magma_int_t m, magma_int_t n, magma_int_t k,
                   const cuDoubleComplex *dV,    magma_int_t ldv,
                   const cuDoubleComplex *dT,    magma_int_t ldt,
                   cuDoubleComplex *dC,          magma_int_t ldc,
                   cuDoubleComplex *dwork,       magma_int_t ldwork )
{
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    if (m <= 0 || n <= 0)
        return MAGMA_SUCCESS;

    // W = C^H V
    // magma_zgemm( MagmaConjTrans, MagmaNoTrans,
    magmablas_zgemm_reduce(
                           n, k, m,
                           c_one,  dC,    ldc,
                           dV,    ldv,
                           c_zero, dwork, ldwork);

    // W = W T^H = C^H V T^H
    magma_ztrmm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                 n, k,
                 c_one, dT,    ldt,
                 dwork, ldwork);

    // C = C - V W^H = C - V T V^H C = (I - V T V^H) C = H C
    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                 m, n, k,
                 c_neg_one, dV,    ldv,
                 dwork, ldwork,
                 c_one,     dC,    ldc);

    return MAGMA_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////

__global__ void
magma_zgemv_kernel1(int m, const cuDoubleComplex * __restrict__ V, int ldv,
                    const cuDoubleComplex * __restrict__ c,
                    cuDoubleComplex *dwork);
__global__ void
magma_zgemv_kernel2(int m, int n, const cuDoubleComplex * __restrict__ V, int ldv,
                    const cuDoubleComplex * __restrict__ x, cuDoubleComplex *c);
__global__ void 
magma_ztrmv_tkernel(cuDoubleComplex *T, int ldt, cuDoubleComplex *v,
                                    cuDoubleComplex *y);
__global__ void
magma_dznrm2_adjust_kernel(double *xnorm, cuDoubleComplex *c);

extern "C" void
magma_zlarfgx_gpu(int n, cuDoubleComplex *dx0, cuDoubleComplex *dx,
                  cuDoubleComplex *dtau, double *dxnorm,
                  cuDoubleComplex *dA, int it);

extern "C" magma_int_t
magma_zgeqr2x4_gpu(magma_int_t *m, magma_int_t *n, cuDoubleComplex *dA, 
                   magma_int_t *ldda, cuDoubleComplex *dtau,
                   cuDoubleComplex *dT, cuDoubleComplex *ddA,
                   double *dwork, magma_int_t *info, cudaStream_t stream)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:   
    A = Q * R.

    This expert routine requires two more arguments than the standard 
    zgeqr2, namely, dT and ddA, explained below. The storage for A is 
    also not as in the LAPACK's zgeqr2 routine (see below). 

    The first is used to output the triangular 
    n x n factor T of the block reflector used in the factorization. 
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    This version adds internal blocking.

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the m by n matrix A.   
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

            the elements on and above the diagonal of the array   
            contain the min(m,n) by n upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the unitary matrix Q as a   
            product of elementary reflectors (see Further Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    dT      (output) COMPLEX_16 array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector 
            used in the factorization. The lower triangular part is 0.

    ddA     (output) COMPLEX_16 array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    RWORK   (workspace) DOUBLE_PRECISION array, dimension (3 N)

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   
    =====================================================================    */

    #define da_ref(a_1,a_2) ( dA+(a_2)*(*ldda) + (a_1))
    #define dt_ref(a_1,a_2) ( dT+(a_2)*(k) + (a_1))
    #define BS 32

    static magma_int_t i, k;

    double *dnorm = (double *)dwork;
    cuDoubleComplex *work = (cuDoubleComplex *)(dwork+2*(*n));

    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);
    magmablasSetKernelStream(stream);

    *info = 0;
    if (*m < 0) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*ldda < max(1,*m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(*m,*n);
    magmablas_dznrm2(*m, k, da_ref(0,0), *ldda, dnorm);

    for (int b=0; b < k; b += BS) {
        for (i = b; i < min(k, b+BS); ++i) {

            /*   Apply H' to A(:,i) from the left                           */    
            if ( i-b > 0){
                magma_zgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>( *m-i+1, da_ref(i-1,0), *ldda,
                                                    da_ref(i-1, i-1), work, dtau+i-1);
                magma_ztrmv_kernel2<<< i-1, i-1, 0, magma_stream >>>( dt_ref(0,0), k, work,
                                                    dt_ref(0,i-1), dtau+i-1);

                /* dwork = V' c                   */
                magma_zgemv_kernel1<<< i-b, BLOCK_SIZE, 0, magma_stream >>>(*m-b, da_ref(b, b), 
                             *ldda, da_ref(b,i), work);

                /* dwork = T' work                */
                magma_ztrmv_tkernel<<< i-b, i-b, 0, magma_stream >>>(dt_ref(b,b), k, work, work+i-b);

                /* c = c - V work                 */
                dim3  blocks3( (*m-b + BLOCK_SIZE-1) / BLOCK_SIZE );
                dim3 threads3( BLOCK_SIZE );
                magma_zgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>(*m-b, i-b, da_ref(b,b), *ldda, 
                                   work+i-b, da_ref(b, i));
            }

            /*   Adjust the dnorm[i] to hold the norm of A(i:m,i)           */ 
            if ( i > 0 )
                magma_dznrm2_adjust_kernel<<< 1, i, 0, magma_stream >>> (dnorm+i, da_ref(0, i));
            
            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) 
                1. 1 is not yet put on the diagonal of A
                2. Elements above the diagonal are copied in ddA and
                   the ones in A are set to zero                                         
                3. update T                                                 */
            magma_zlarfgx_gpu(*m-i, da_ref(i, i), da_ref(min(i+1,*m),i), dtau+i, 
                              dnorm+i, ddA + i + i*(*n), i);

            if (i==0){
              cuDoubleComplex tt = MAGMA_Z_ONE;
              magmablas_zlacpy(MagmaUpperLower, 1, 1, dtau, 1, dt_ref(0,0), 1);
              magma_zsetmatrix(1,1, &tt,1, da_ref(i, i),1);
            }
/*
            else
             {
                // Compute the i-th column of T.
                //   Set da_ref(i, i) = 1.                                    
                magma_zgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>( *m-i, da_ref(i,0), *ldda, 
                                          da_ref(i, i), work, dtau+i);
                magma_ztrmv_kernel2<<< i, i, 0, magma_stream          >>>( dt_ref(0,0), k, work, 
                                                          dt_ref(0,i), dtau+i);
              }
*/

        }
        magma_zgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>( *m-i+1, da_ref(i-1,0), *ldda,
                                                    da_ref(i-1, i-1), work, dtau+i-1);
        magma_ztrmv_kernel2<<< i-1, i-1, 0, magma_stream >>>( dt_ref(0,0), k, work,
                                                    dt_ref(0,i-1), dtau+i-1);

        
        /* Apply the transformations to the trailing matrix. */
        //magma_zlarfb2_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
        magma_zlarfb2_gpu(
                           *m-b, k-i, BS,
                           da_ref(b, b), *ldda, dT+b+b*k, k,
                           da_ref(b, i), *ldda, work, k-i);
    }

    magmablasSetKernelStream(cstream);

    return *info;
} /* magma_zgeqr2 */
