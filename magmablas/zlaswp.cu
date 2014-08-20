/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       
       @author Stan Tomov
       @author Mathieu Faverge
       @author Ichitaro Yamazaki
       @author Mark Gates
*/
#include "common_magma.h"

// MAX_PIVOTS is maximum number of pivots to apply in each kernel launch
// NTHREADS is number of threads in a block
// 64 and 256 are better on Kepler; 
//#define MAX_PIVOTS 64
//#define NTHREADS   256
#define MAX_PIVOTS 32
#define NTHREADS   64

typedef struct {
    magmaDoubleComplex *dAT;
    int n, lda, j0, npivots;
    int ipiv[MAX_PIVOTS];
} zlaswp_params_t;


// Matrix A is stored row-wise in dAT.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswp_kernel( zlaswp_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int lda = params.lda;
        magmaDoubleComplex *dAT = params.dAT + tid + params.j0*lda;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dAT + i2*lda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}


// Launch zlaswp kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void zlaswp_launch( zlaswp_params_t &params, magma_queue_t queue )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    zlaswp_kernel<<< blocks, NTHREADS, 0, queue >>>( params );
}


// @deprecated
// Swap rows of A, stored row-wise.
// This version updates each entry of ipiv by adding ind.
// (In contrast, LAPACK applies laswp, then updates ipiv.)
// It is used in zgetrf, zgetrf_gpu, zgetrf_mgpu, zgetrf_ooc.
extern "C" void
magmablas_zpermute_long2( magma_int_t n, magmaDoubleComplex *dAT, magma_int_t lda,
                          magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n  lda  j0       npivots
        zlaswp_params_t params = { dAT, n, lda, ind + k, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1;
            ipiv[ind + k + j] += ind;
        }
        zlaswp_launch( params, magma_stream );
    }
}


// @deprecated
// Swap rows of A, stored row-wise.
// This version assumes ind has already been added to ipiv.
// (In contrast, LAPACK applies laswp, then updates ipiv.)
// It is used in zgetrf_mgpu, zgetrf_ooc.
extern "C" void
magmablas_zpermute_long3( magmaDoubleComplex *dAT, magma_int_t lda,
                          const magma_int_t *ipiv, magma_int_t nb, magma_int_t ind )
{
    for( int k = 0; k < nb; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, nb-k );
        // fields are:             dAT  n    lda  j0       npivots
        zlaswp_params_t params = { dAT, lda, lda, ind + k, npivots };
        for( int j = 0; j < MAX_PIVOTS; ++j ) {
            params.ipiv[j] = ipiv[ind + k + j] - k - 1 - ind;
        }
        zlaswp_launch( params, magma_stream );
    }
}


/**
    Purpose:
    =============
    ZLASWP performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dAT      COMPLEX*16 array on GPU, stored row-wise, dimension (LDA,N)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    lda      INTEGER
             The leading dimension of the array A. lda >= n.
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
// It is used in zgessm, zgetrf_incpiv.
extern "C" void
magmablas_zlaswp_q(
    magma_int_t n, magmaDoubleComplex *dAT, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 || k1 >= lda )
        info = -4;
    else if ( k2 < 0 || k2 >= lda || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        // fields are:             dAT        n  lda  j0 npivots
        zlaswp_params_t params = { dAT+k*lda, n, lda, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswp_launch( params, queue );
    }
}


/**
    @see magmablas_zlaswp_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp( magma_int_t n, magmaDoubleComplex *dAT, magma_int_t lda,
                  magma_int_t k1, magma_int_t k2,
                  const magma_int_t *ipiv, magma_int_t inci )
{
    magmablas_zlaswp_q( n, dAT, lda, k1, k2, ipiv, inci, magma_stream );
}






// ------------------------------------------------------------
// Extended version has stride in both directions (ldx, ldy)
// to handle both row-wise and column-wise storage.

typedef struct {
    magmaDoubleComplex *dA;
    int n, ldx, ldy, j0, npivots;
    int ipiv[MAX_PIVOTS];
} zlaswpx_params_t;


// Matrix A is stored row or column-wise in dA.
// Divide matrix A into block-columns of NTHREADS columns each.
// Each GPU block processes one block-column of A.
// Each thread goes down a column of A,
// swapping rows according to pivots stored in params.
__global__ void zlaswpx_kernel( zlaswpx_params_t params )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < params.n ) {
        int ldx = params.ldx;
        magmaDoubleComplex *dA = params.dA + tid*params.ldy + params.j0*ldx;
        magmaDoubleComplex *A1  = dA;
        
        for( int i1 = 0; i1 < params.npivots; ++i1 ) {
            int i2 = params.ipiv[i1];
            magmaDoubleComplex *A2 = dA + i2*ldx;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += ldx;  // A1 = dA + i1*ldx
        }
    }
}


// Launch zlaswpx kernel with ceil( n / NTHREADS ) blocks of NTHREADS threads each.
extern "C" void zlaswpx( zlaswpx_params_t &params, magma_queue_t queue )
{
    int blocks = (params.n + NTHREADS - 1) / NTHREADS;
    zlaswpx_kernel<<< blocks, NTHREADS, 0, queue >>>( params );
}


/**
    Purpose:
    =============
    ZLASWPX performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored either row-wise or column-wise,
       depending on ldx and ldy. **
    Otherwise, this is identical to LAPACK's interface.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dA       COMPLEX*16 array on GPU, dimension (*,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    ldx      INTEGER
             Stride between elements in same column.
    
    \param[in]
    ldy      INTEGER
             Stride between elements in same row.
             For A stored row-wise,    set ldx=lda and ldy=1.
             For A stored column-wise, set ldx=1   and ldy=lda.
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    ipiv     INTEGER array, on CPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswpx_q(
    magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;  
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    for( int k = k1-1; k < k2; k += MAX_PIVOTS ) {
        int npivots = min( MAX_PIVOTS, k2-k );
        // fields are:              dA        n  ldx  ldy  j0 npivots
        zlaswpx_params_t params = { dA+k*ldx, n, ldx, ldy, 0, npivots };
        for( int j = 0; j < npivots; ++j ) {
            params.ipiv[j] = ipiv[(k+j)*inci] - k - 1;
        }
        zlaswpx( params, queue );
    }
}


/**
    @see magmablas_zlaswpx_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswpx( magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldx, magma_int_t ldy,
                   magma_int_t k1, magma_int_t k2,
                   const magma_int_t *ipiv, magma_int_t inci )
{
    return magmablas_zlaswpx_q( n, dA, ldx, ldy, k1, k2, ipiv, inci, magma_stream );
}






// ------------------------------------------------------------
// This version takes d_ipiv on the GPU. Thus it does not pass pivots
// as an argument using a structure, avoiding all the argument size
// limitations of CUDA and OpenCL. It also needs just one kernel launch
// with all the pivots, instead of multiple kernel launches with small
// batches of pivots. On Fermi, it is faster than magmablas_zlaswp
// (including copying pivots to the GPU).

__global__ void zlaswp2_kernel(
    int n, magmaDoubleComplex *dAT, int lda, int npivots,
    const magma_int_t* d_ipiv, magma_int_t inci )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    if( tid < n ) {
        dAT += tid;
        magmaDoubleComplex *A1  = dAT;
        
        for( int i1 = 0; i1 < npivots; ++i1 ) {
            int i2 = d_ipiv[i1*inci] - 1;  // Fortran index
            magmaDoubleComplex *A2 = dAT + i2*lda;
            magmaDoubleComplex temp = *A1;
            *A1 = *A2;
            *A2 = temp;
            A1 += lda;  // A1 = dA + i1*ldx
        }
    }
}


/**
    Purpose:
    =============
    ZLASWP2 performs a series of row interchanges on the matrix A.
    One row interchange is initiated for each of rows K1 through K2 of A.
    
    ** Unlike LAPACK, here A is stored row-wise (hence dAT). **
    Otherwise, this is identical to LAPACK's interface.
    
    Here, d_ipiv is passed in GPU memory.
    
    Arguments:
    ==========
    \param[in]
    n        INTEGER
             The number of columns of the matrix A.
    
    \param[in,out]
    dAT      COMPLEX*16 array on GPU, stored row-wise, dimension (LDA,*)
             On entry, the matrix of column dimension N to which the row
             interchanges will be applied.
             On exit, the permuted matrix.
    
    \param[in]
    lda      INTEGER
             The leading dimension of the array A.
             (I.e., stride between elements in a column.)
    
    \param[in]
    k1       INTEGER
             The first element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    k2       INTEGER
             The last element of IPIV for which a row interchange will
             be done. (One based index.)
    
    \param[in]
    d_ipiv   INTEGER array, on GPU, dimension (K2*abs(INCI))
             The vector of pivot indices.  Only the elements in positions
             K1 through K2 of IPIV are accessed.
             IPIV(K) = L implies rows K and L are to be interchanged.
    
    \param[in]
    inci     INTEGER
             The increment between successive values of IPIV.
             Currently, IPIV > 0.
             TODO: If IPIV is negative, the pivots are applied in reverse order.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp2_q(
    magma_int_t n, magmaDoubleComplex* dAT, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *d_ipiv, magma_int_t inci,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( k1 < 0 )
        info = -4;  
    else if ( k2 < 0 || k2 < k1 )
        info = -5;
    else if ( inci <= 0 )
        info = -7;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }
    
    int blocks = (n + NTHREADS - 1) / NTHREADS;
    zlaswp2_kernel<<< blocks, NTHREADS, 0, queue >>>(
        n, dAT + (k1-1)*lda, lda, k2-(k1-1), d_ipiv, inci );
}


/**
    @see magmablas_zlaswp2_q
    @ingroup magma_zaux2
    ********************************************************************/
extern "C" void
magmablas_zlaswp2( magma_int_t n, magmaDoubleComplex* dAT, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   const magma_int_t *d_ipiv, magma_int_t inci )
{
    magmablas_zlaswp2_q( n, dAT, lda, k1, k2, d_ipiv, inci, magma_stream );
}
