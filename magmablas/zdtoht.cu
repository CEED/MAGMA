/*
    -- MAGMA (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2010

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#define PRECISION_z
#include "commonblas.h"

//
//	m, n - dimensions in the output (ha) matrix.
//             This routine copies the dat matrix from the GPU
//             to ha on the CPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zdtoht(cuDoubleComplex *dat, int ldda,
                 cuDoubleComplex  *ha, int lda,
                 cuDoubleComplex  *dB, int lddb,
                 int m, int n , int nb)
{
    int i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in zdtoht.\n");
	return;
    }

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);

    for(i=0; i<n; i+=nb){
       /* Move data from GPU to CPU using two buffers; first transpose the data on the GPU */
       ib   = min(n-i, nb);
       cudaStreamSynchronize(stream[j%2]);
       magmablas_ztranspose2( dB + (j%2)*nb*lddb, lddb, dat+i, ldda, ib, m);
       cudaMemcpy2DAsync(ha+i*lda, lda*sizeof(cuDoubleComplex),
                         dB + (j%2) * nb * lddb, lddb*sizeof(cuDoubleComplex),
                         sizeof(cuDoubleComplex)*m, ib, 
                         cudaMemcpyDeviceToHost, stream[j%2]);
       j++;
    }

    cudaStreamDestroy( stream[0] );
    cudaStreamDestroy( stream[1] );
}

