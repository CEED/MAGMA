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
//	m, n - dimensions in the source (input) matrix.
//             This routine copies the ha matrix from the CPU
//             to dat on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m). 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zhtodt(cuDoubleComplex  *ha, int lda, 
                 cuDoubleComplex *dat, int ldda,
                 cuDoubleComplex  *dB, int lddb,
                 int m, int n , int nb)
{
    int i = 0, j = 0, ib;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || ldda < n || lddb < m){
        printf("Wrong arguments in zhotodt.\n");
	return;
    }

    static cudaStream_t stream[2];
    cudaStreamCreate(&stream[0]);
    cudaStreamCreate(&stream[1]);
   
    /* Move data from CPU to GPU in the first panel in the dB buffer */
    ib   = min(n-i, nb);
    cudaMemcpy2DAsync(dB + (j%2) * nb * lddb, lddb*sizeof(cuDoubleComplex),
                      ha + i*lda, lda*sizeof(cuDoubleComplex),
                      sizeof(cuDoubleComplex)*m, ib,
                      cudaMemcpyHostToDevice, stream[j%2]);
    j++;

    for(i=nb; i<n; i+=nb){
       /* Move data from CPU to GPU in the second panel in the dB buffer */
       ib   = min(n-i, nb);
       cudaMemcpy2DAsync(dB + (j%2) * nb * lddb, lddb*sizeof(cuDoubleComplex),
                         ha+i*lda, lda*sizeof(cuDoubleComplex),
                         sizeof(cuDoubleComplex)*m, ib, 
                         cudaMemcpyHostToDevice, stream[j%2]);
       j++;
  
       /* Make sure that the previous panel (i.e., j%2) has arrived 
          and transpose it directly into the dat matrix                  */
       cudaStreamSynchronize(stream[j%2]);
       magmablas_ztranspose2( dat+i-nb, ldda, dB + (j%2)*nb*lddb, lddb, m, nb);
    }

    /* Transpose the last part of the matrix.                            */
    j++;
    cudaStreamSynchronize(stream[j%2]);
    magmablas_ztranspose2( dat+i-nb, ldda, dB + (j%2)*nb*lddb, lddb, m, ib);

    cudaStreamDestroy( stream[0] );
    cudaStreamDestroy( stream[1] );
}

