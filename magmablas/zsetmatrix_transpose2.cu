/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#define PRECISION_z
#include "commonblas.h"

//
//    m, n - dimensions in the source (input) matrix.
//             This routine copies the ha matrix from the CPU
//             to dat on the GPU. In addition, the output matrix
//             is transposed. The routine uses a buffer of size
//             2*lddb*nb pointed to by dB (lddb > m) on the GPU. 
//             Note that lda >= m and lddat >= n.
//
extern "C" void 
magmablas_zsetmatrix_transpose3(
                  magma_int_t num_gpus, cudaStream_t **stream0,
                  cuDoubleComplex  *ha,  int lda, 
                  cuDoubleComplex **dat, int ldda, int starti,
                  cuDoubleComplex **dB,  int lddb,
                  int m, int n , int nb)
{
    int d, i = 0, offset = 0, j = 0, j_local, ib;
    static cudaStream_t stream[4][2];

    /* Quick return */
    if ( (m == 0) || (n == 0) )
        return;

    if (lda < m || num_gpus*ldda < n || lddb < m){
        printf("Wrong arguments in zhtodt2 (%d<%d), (%d*%d<%d), or (%d<%d).\n",lda,m,num_gpus,ldda,n,lddb,m);
        return;
    }
    
    for( d=0; d<num_gpus; d++ ) {
      cudaSetDevice(d);
      cudaStreamCreate(&stream[d][0]);
      cudaStreamCreate(&stream[d][1]);
    }

    /* Move data from CPU to GPU in the first panel in the dB buffer */
    for( d=0; d<num_gpus; d++ ) {
      cudaSetDevice(d);
      i  = d * nb;
      ib = min(n-i, nb);

      //printf( " sent A(:,%d:%d) to %d-th panel of %d gpu\n",i,i+ib-1,0,d );
      cudaMemcpy2DAsync(dB[d],     lddb*sizeof(cuDoubleComplex),
                        ha + i*lda, lda*sizeof(cuDoubleComplex),
                        sizeof(cuDoubleComplex)*m, ib,
                        cudaMemcpyHostToDevice, stream[d][0]);
      j++;
    }

    for(i=num_gpus*nb; i<n; i+=nb){
       /* Move data from CPU to GPU in the second panel in the dB buffer */
       d       = j%num_gpus;
       j_local = j/num_gpus;
       cudaSetDevice(d);


       ib = min(n-i, nb);
       //printf( " sent A(:,%d:%d) to %d-th panel of %d gpu (%d,%d)\n",i,i+ib-1,j_local%2,d,m,ib );
       cudaMemcpy2DAsync(dB[d] + (j_local%2) * nb * lddb, lddb*sizeof(cuDoubleComplex),
                         ha+i*lda, lda*sizeof(cuDoubleComplex),
                         sizeof(cuDoubleComplex)*m, ib, 
                         cudaMemcpyHostToDevice, stream[d][j_local%2]);
  
       /* Make sure that the previous panel (i.e., j%2) has arrived 
          and transpose it directly into the dat matrix                  */
       j_local = (j-num_gpus)/num_gpus;
       //printf( " wait for %d-th panel of %d gpu and store in %d:%d\n\n",j_local%2,d,nb*(starti+j_local),nb*(starti+j_local)+nb-1 );
       cudaStreamSynchronize(stream[d][j_local%2]);
       magmablas_ztranspose2(dat[d]+nb*(starti+j_local), ldda, 
                             dB[d] +(j_local%2)*nb*lddb, lddb, 
                             m, nb);
       j++;
       offset += nb;
    }

    /* Transpose the last part of the matrix.                            */
    for( i=0; i<num_gpus; i++ ) {
      d       = j%num_gpus;
      j_local = (j-num_gpus)/num_gpus;
      ib      = min(n-offset, nb);

      cudaSetDevice(d);

      //printf( " wait for %d-th panel of %d gpu and store in %d:%d ((%d+%d)*%d)\n\n",j_local%2,d,nb*(starti+j_local),nb*(starti+j_local)+ib-1,starti,j_local,nb );
      cudaStreamSynchronize(stream[d][j_local%2]);
      magmablas_ztranspose2(dat[d]+nb*(starti+j_local), ldda, 
                            dB[d] +(j_local%2)*nb*lddb, lddb, 
                            m, ib);
      j++;
      offset += nb;
    }

    
    for( d=0; d<num_gpus; d++ ) {
      cudaSetDevice(d);
      cudaStreamDestroy( stream[d][0] );
      cudaStreamDestroy( stream[d][1] );
    }
    
}

