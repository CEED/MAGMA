/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Ichitaro Yamazaki
*/
#include "common_magma.h"

#define PRECISION_z

//
//    m, n - dimensions in the source (input) matrix.
//             This routine copies the hA matrix from the CPU
//             to dAT on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddw*nb pointed to by dwork (lddw > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zsetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const magmaDoubleComplex *hA,       magma_int_t lda, 
    magmaDoubleComplex_ptr    dAT[],    magma_int_t ldda, 
    magmaDoubleComplex_ptr    dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb)
{
#define hA(j)       (hA         + (j)*lda)
#define dwork(d, j) (dwork[(d)] + (j)*nb*lddw)
#define dAT(d, j)   (dAT[(d)]   + (j)*nb)

    magma_int_t nqueues = 2, d, j, j_local, id, ib;
    
    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;
    
    if (lda < m || ngpu*ldda < n || lddw < m) {
        printf( "Wrong arguments in magmablas_zsetmatrix_transpose_mgpu (%d<%d), (%d*%d<%d), or (%d<%d).\n",
                (int) lda, (int) m, (int) ngpu, (int) ldda, (int) n, (int) lddw, (int) m );
        return;
    }
    
    /* Move data from CPU to GPU by block columns and transpose it */
    for (j=0; j < n; j += nb) {
        d       = (j/nb)%ngpu;
        j_local = (j/nb)/ngpu;
        id      = j_local%nqueues;
        magma_setdevice(d);
        
        ib = min(n-j, nb);
        magma_zsetmatrix_async( m, ib,
                                hA(j),        lda,
                                dwork(d, id), lddw, 
                                queues[d][id] );
        
        magmablas_ztranspose_q( m, ib, dwork(d,id), lddw, dAT(d,j_local), ldda, queues[d][id] );
    }
}
