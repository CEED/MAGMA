/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Ichitaro Yamazaki
*/
#include "common_magma.h"
#define PRECISION_z
#include "commonblas.h"

//
//    m, n - dimensions in the output (ha) matrix.
//             This routine copies the dat matrix from the GPU
//             to ha on the CPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zgetmatrix_transpose_mgpu(
                  magma_int_t ngpus, magma_queue_t stream[][2],
                  magmaDoubleComplex **dat, magma_int_t ldda,
                  magmaDoubleComplex   *ha, magma_int_t lda,
                  magmaDoubleComplex  **db, magma_int_t lddb,
                  magma_int_t m, magma_int_t n, magma_int_t nb)
{
#define   A(j)     (ha  + (j)*lda)
#define  dB(d, j)  (db[(d)]  + (j)*nb*lddb)
#define  dAT(d, j) (dat[(d)] + (j)*nb)
    int nstreams = 2, j, j_local, d, id, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ngpus*ldda < n || lddb < m){
        printf( "Wrong arguments in magmablas_zgetmatrix_transpose_mgpu (%d<%d), (%d*%d<%d), or (%d<%d).\n",
                lda, m, ngpus, ldda, n, lddb, m );
        return;
    }
    
    /* Move data from GPU to CPU using two buffers; first transpose the data on the GPU */
    for(j=0; j<n; j+=nb){
       d       = (j/nb)%ngpus;
       j_local = (j/nb)/ngpus;
       id      = j_local%nstreams;
       magma_setdevice(d);

       ib = min(n-j, nb);
       magmablasSetKernelStream(stream[d][id]);
       magmablas_ztranspose2(  dB(d, id),      lddb, 
                              dAT(d, j_local), ldda, 
                              ib, m);
       magma_zgetmatrix_async( m, ib,
                               dB(d, id), lddb,
                               A(j),      lda, 
                               stream[d][id] );
    }
}



