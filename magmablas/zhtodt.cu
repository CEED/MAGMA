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
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
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
        printf("Wrong arguments in zhtodt.\n");
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

//===========================================================================
//  This version is similar to the above but for multiGPUs. The distribution
//  is 1D block cyclic. The input arrays are pointers for the corresponding 
//  GPUs. The streams are passed as argument, in contrast to the single GPU
//  routine.
//===========================================================================
extern "C" void 
magmablas_zhtodt2(int num_gpus, cudaStream_t stream[][2],
                  cuDoubleComplex  *ha,  int  lda, 
                  cuDoubleComplex **dat, int *ldda,
                  cuDoubleComplex  **dB, int lddb,
                  int m, int n , int nb)
{
    int i = 0, j[4] = {0, 0, 0, 0}, ib, k = 0;

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || lddb < m){
        printf("Wrong arguments in zhtodt2.\n");
	return;
    }

    if (n<num_gpus*nb){
       for(i=0; i<n; i+=nb){
          k = (i/nb)%num_gpus;
          cudaSetDevice(k);

          ib = min(n-i, nb);
          cudaMemcpy2DAsync(dB[k], lddb*sizeof(cuDoubleComplex),
                            ha+i*lda, lda*sizeof(cuDoubleComplex),
                            sizeof(cuDoubleComplex)*m, ib,
                            cudaMemcpyHostToDevice, stream[k][0]);
       }
       for(i=0; i<n; i+=nb){
          k = (i/nb)%num_gpus;
          cudaSetDevice(k);

          ib = min(n-i, nb);
          cudaStreamSynchronize( stream[k][0]);
          ib = min(n-i, nb);
          magmablas_ztranspose2( dat[k]+ i/(nb*num_gpus)*nb, ldda[k],
                                 dB[k], lddb, m, ib);
       }
    } 
    else
    {
      for(i=0; i<(n + num_gpus*nb); i+=nb){
         k = (i/nb)%num_gpus;
         cudaSetDevice(k);

         if (i<n){
            /* Move data from CPU to GPU in the second panel in the dB buffer */
            ib = min(n-i, nb);
            cudaMemcpy2DAsync(dB[k] + (j[k]%2)*nb*lddb, lddb*sizeof(cuDoubleComplex),
                         ha+i*lda, lda*sizeof(cuDoubleComplex),
                         sizeof(cuDoubleComplex)*m, ib, 
                         cudaMemcpyHostToDevice, stream[k][j[k]%2]);
         }
         j[k]++;
  
         if (i> (num_gpus-1)*nb){
            /* Make sure that the previous panel (i.e., j[k]%2) has arrived 
               and transpose it directly into the dat matrix                  */
            cudaStreamSynchronize( stream[k][ j[k]%2 ]);
            ib = min(n - i + num_gpus*nb, nb);
            magmablas_ztranspose2( dat[k]+ i/(nb*num_gpus)*nb -nb, ldda[k],
                                   dB[k] +(j[k]%2)*nb*lddb, lddb, m, ib);
         }
      }
    }
}
