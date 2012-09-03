/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Raffaele Solca

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "trace.h"

#include <cblas.h>
#include <assert.h>

#define PRECISION_z

#define MAGMABLAS_ZHEMV_MGPU
#ifdef  MAGMABLAS_ZHEMV_MGPU
#define magmablas_zhemv_200_mgpu_offset magmablas_zhemv_mgpu_offset
#define magmablas_zhemv_200_mgpu_32_offset magmablas_zhemv_mgpu_32_offset

extern "C"
magma_int_t
magmablas_zhemv_200_mgpu_offset( char uplo, magma_int_t n,
                                 cuDoubleComplex alpha,
                                 cuDoubleComplex **A, magma_int_t lda,
                                 cuDoubleComplex **X, magma_int_t incx,
                                 cuDoubleComplex beta,
                                 cuDoubleComplex **Y, magma_int_t incy,
                                 cuDoubleComplex **work, magma_int_t lwork,
                                 magma_int_t num_gpus,
                                 magma_int_t nb,
                                 magma_int_t offset,
                                 cudaStream_t stream[][10]);

extern "C"
magma_int_t
magmablas_zhemv_200_mgpu_32_offset( char uplo, magma_int_t n,
                                    cuDoubleComplex alpha,
                                    cuDoubleComplex **A, magma_int_t lda,
                                     cuDoubleComplex **X, magma_int_t incx,
                                    cuDoubleComplex beta,
                                    cuDoubleComplex **Y, magma_int_t incy,
                                    cuDoubleComplex **work, magma_int_t lwork,
                                      magma_int_t num_gpus,
                                    magma_int_t nb,
                                    magma_int_t offset,
                                    cudaStream_t stream[][10]);
#endif
extern "C"
magma_int_t
magmablas_zhemv_mgpu( magma_int_t num_gpus, magma_int_t k, char uplo, 
                      magma_int_t n, magma_int_t nb,
                      cuDoubleComplex alpha,
                      cuDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
                      cuDoubleComplex **dx, magma_int_t incx,
                      cuDoubleComplex beta,
                      cuDoubleComplex **dy, magma_int_t incy, 
                      cuDoubleComplex **dwork, magma_int_t ldwork,
                      cuDoubleComplex *work, cuDoubleComplex *w,
                      cudaStream_t stream[][10]);

extern "C"
magma_int_t
magmablas_zhemv_synch( magma_int_t num_gpus, magma_int_t k, char uplo, 
                      magma_int_t n, cuDoubleComplex *work, cuDoubleComplex *w,
                      cudaStream_t stream[][10]);

#define A(i, j) (a+(j)*lda + (i))
#define W(i, j) (w+(j)*ldw + (i))

#define dA(id, i, j) (da[(id)]+((j)+loffset)*ldda + (i)+offset)
#define dW(id, i, j)  (dw[(id)]+  (j)      *lddw + (i))

#define dW1(id, i, j) (dw[(id)]+ ((j)+nb0) *lddw + (i))

#ifdef __cplusplus
extern "C" {
#endif
    magma_int_t
    magmablas_zhemv( char uplo, magma_int_t n,
                     cuDoubleComplex alpha,
                     cuDoubleComplex *A, magma_int_t lda,
                     cuDoubleComplex *X, magma_int_t incx,
                     cuDoubleComplex beta,
                     cuDoubleComplex *Y, magma_int_t incy);
#define magma_zhemv magmablas_zhemv
#define magma_zgemv magmablas_zgemv
#ifdef __cplusplus
}
#endif
extern "C" void
magmablas_zgemvt(char flag, int m, int n, cuDoubleComplex alpha,
                 cuDoubleComplex *A, int lda, cuDoubleComplex *x, int incx, cuDoubleComplex beta, cuDoubleComplex *y, int incy);


extern "C" double
magma_zlatrd_mgpu(int num_gpus, char uplo, magma_int_t n, magma_int_t nb, magma_int_t nb0,
                  cuDoubleComplex *a,  magma_int_t lda, 
                  double *e, cuDoubleComplex *tau, 
                  cuDoubleComplex *w,   magma_int_t ldw,
                  cuDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
                  cuDoubleComplex **dw, magma_int_t lddw, 
                  cuDoubleComplex *dwork[4], magma_int_t ldwork,
                  magma_int_t k, 
                  cuDoubleComplex  *dx[4], cuDoubleComplex *dy[4], 
                  cuDoubleComplex *work,
                  cudaStream_t stream[][10] )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   
    ZLATRD reduces NB rows and columns of a complex Hermitian matrix A to   
    Hermitian tridiagonal form by an orthogonal similarity   
    transformation Q' * A * Q, and returns the matrices V and W which are   
    needed to apply the transformation to the unreduced part of A.   

    If UPLO = 'U', ZLATRD reduces the last NB rows and columns of a   
    matrix, of which the upper triangle is supplied;   
    if UPLO = 'L', ZLATRD reduces the first NB rows and columns of a   
    matrix, of which the lower triangle is supplied.   

    This is an auxiliary routine called by ZHETRD.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            Specifies whether the upper or lower triangular part of the   
            Hermitian matrix A is stored:   
            = 'U': Upper triangular   
            = 'L': Lower triangular   

    N       (input) INTEGER   
            The order of the matrix A.   

    NB      (input) INTEGER   
            The number of rows and columns to be reduced.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading   
            n-by-n upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading n-by-n lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   
            On exit:   
            if UPLO = 'U', the last NB columns have been reduced to   
              tridiagonal form, with the diagonal elements overwriting   
              the diagonal elements of A; the elements above the diagonal   
              with the array TAU, represent the orthogonal matrix Q as a   
              product of elementary reflectors;   
            if UPLO = 'L', the first NB columns have been reduced to   
              tridiagonal form, with the diagonal elements overwriting   
              the diagonal elements of A; the elements below the diagonal   
              with the array TAU, represent the  orthogonal matrix Q as a   
              product of elementary reflectors.   
            See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= (1,N).   

    E       (output) COMPLEX_16 array, dimension (N-1)   
            If UPLO = 'U', E(n-nb:n-1) contains the superdiagonal   
            elements of the last NB columns of the reduced matrix;   
            if UPLO = 'L', E(1:nb) contains the subdiagonal elements of   
            the first NB columns of the reduced matrix.   

    TAU     (output) COMPLEX_16 array, dimension (N-1)   
            The scalar factors of the elementary reflectors, stored in   
            TAU(n-nb:n-1) if UPLO = 'U', and in TAU(1:nb) if UPLO = 'L'.   
            See Further Details.   

    W       (output) COMPLEX_16 array, dimension (LDW,NB)   
            The n-by-nb matrix W required to update the unreduced part   
            of A.   

    LDW     (input) INTEGER   
            The leading dimension of the array W. LDW >= max(1,N).   

    Further Details   
    ===============   
    If UPLO = 'U', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(n) H(n-1) . . . H(n-nb+1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(i:n) = 0 and v(i-1) = 1; v(1:i-1) is stored on exit in A(1:i-1,i),   
    and tau in TAU(i-1).   

    If UPLO = 'L', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(1) H(2) . . . H(nb).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i) = 0 and v(i+1) = 1; v(i+1:n) is stored on exit in A(i+1:n,i),   
    and tau in TAU(i).   

    The elements of the vectors v together form the n-by-nb matrix V   
    which is needed, with W, to apply the transformation to the unreduced   
    part of the matrix, using a Hermitian rank-2k update of the form:   
    A := A - V*W' - W*V'.   

    The contents of A on exit are illustrated by the following examples   
    with n = 5 and nb = 2:   

    if UPLO = 'U':                       if UPLO = 'L':   

      (  a   a   a   v4  v5 )              (  d                  )   
      (      a   a   v4  v5 )              (  1   d              )   
      (          a   1   v5 )              (  v1  1   a          )   
      (              d   1  )              (  v1  v2  a   a      )   
      (                  d  )              (  v1  v2  a   a   a  )   

    where d denotes a diagonal element of the reduced matrix, a denotes   
    an element of the original matrix that is unchanged, and vi denotes   
    an element of the vector defining H(i).   
    =====================================================================    */
  
    char uplo_[2]  = {uplo, 0};

    double mv_time = 0.0;
    magma_int_t i;
    magma_int_t loffset = nb0*((offset/nb0)/num_gpus);
  
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magma_int_t id, idw;
    //cuDoubleComplex *work;

    cuDoubleComplex value = MAGMA_Z_ZERO;
    
    magma_int_t kk;  
    magma_int_t ione = 1;

    magma_int_t i_n, i_1, iw;
  
    cuDoubleComplex alpha;
    cuDoubleComplex *f;

    if (n <= 0) {
      return 0;
    }

    magma_zmalloc_cpu( &f, n );
    assert( f != NULL );  // TODO return error, or allocate outside zlatrd
    
    /*cudaStream_t stream[4][10];
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        //magma_zmalloc( &dx[id], k*n );
        //magma_zmalloc( &dy[id], k*n );
        for( kk=0; kk<k; kk++ ) magma_queue_create( &stream[id][kk] );
    }*/
    //magma_zmalloc_pinned( &work, k*num_gpus*n );

    if (lapackf77_lsame(uplo_, "U")) {

      /* Reduce last NB columns of upper triangle */
      for (i = n-1; i >= n - nb ; --i) {
        i_1 = i + 1;
        i_n = n - i - 1;
        
        iw = i - n + nb;
        if (i < n-1) {
          /* Update A(1:i,i) */
          #if defined(PRECISION_z) || defined(PRECISION_c)
              lapackf77_zlacgv(&i_n, W(i, iw+1), &ldw);
          #endif
          blasf77_zgemv("No transpose", &i_1, &i_n, &c_neg_one, A(0, i+1), &lda,
                        W(i, iw+1), &ldw, &c_one, A(0, i), &ione);
          #if defined(PRECISION_z) || defined(PRECISION_c)
              lapackf77_zlacgv(&i_n, W(i, iw+1), &ldw);
              lapackf77_zlacgv(&i_n, A(i, i+1), &ldw);
          #endif
          blasf77_zgemv("No transpose", &i_1, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                        A(i, i+1), &lda, &c_one, A(0, i), &ione);
          #if defined(PRECISION_z) || defined(PRECISION_c)
              lapackf77_zlacgv(&i_n, A(i, i+1), &ldw);
          #endif
        }
        if (i > 0) {
          /* Generate elementary reflector H(i) to annihilate A(1:i-2,i) */
          
          alpha = *A(i-1, i);
          
          lapackf77_zlarfg(&i, &alpha, A(0, i), &ione, &tau[i - 1]);
          
          e[i-1] = MAGMA_Z_REAL( alpha );
          MAGMA_Z_SET2REAL(*A(i-1, i), 1.);
          
          /* Compute W(1:i-1,i) */
          // 1. Send the block reflector  A(0:n-i-1,i) to the GPU
          for( id=0; id<num_gpus; id++ ) {
              magma_setdevice(id);
              magma_zsetvector( i, A(0, i), 1, dA(id, 0, i), 1 );
              magma_zhemv(MagmaUpper, i, c_one, dA(id, 0, 0), ldda,
                          dA(id, 0, i), ione, c_zero, dW(id, 0, iw), ione);
          }
          
          // 2. Start putting the result back (asynchronously)
          magma_zgetmatrix_async( i, 1,
                                  dW(id, 0, iw),     lddw,
                                  W(0, iw) /*test*/, ldw, stream[idw][0] );
          
          if (i < n-1) {
            blasf77_zgemv(MagmaConjTransStr, &i, &i_n, &c_one, W(0, iw+1), &ldw,
                          A(0, i), &ione, &c_zero, W(i+1, iw), &ione);
          }
          
            // 3. Here is where we need it // TODO find the right place
            magma_queue_sync( stream[idw][0] );

          if (i < n-1) {
          
            blasf77_zgemv("No transpose", &i, &i_n, &c_neg_one, A(0, i+1), &lda,
                          W(i+1, iw), &ione, &c_one, W(0, iw), &ione);

            blasf77_zgemv(MagmaConjTransStr, &i, &i_n, &c_one, A(0, i+1), &lda,
                          A(0, i), &ione, &c_zero, W(i+1, iw), &ione);

            blasf77_zgemv("No transpose", &i, &i_n, &c_neg_one, W(0, iw+1), &ldw,
                          W(i+1, iw), &ione, &c_one, W(0, iw), &ione);
          }

          blasf77_zscal(&i, &tau[i - 1], W(0, iw), &ione);

          #if defined(PRECISION_z) || defined(PRECISION_c)
          cblas_zdotc_sub( i, W(0,iw), ione, A(0,i), ione, &value );
          #else
          value = cblas_zdotc( i, W(0,iw), ione, A(0,i), ione );
          #endif
          alpha = tau[i - 1] * -0.5f * value;
          blasf77_zaxpy(&i, &alpha, A(0, i), &ione,
                        W(0, iw), &ione);
        }
      }

    } else {
      /*  Reduce first NB columns of lower triangle */
//#define PROFILE_SYMV
#ifdef  PROFILE_SYMV
      cudaEvent_t start, stop;
      float etime;
      magma_setdevice(0);
      magma_event_create( &start );
      magma_event_create( &stop  );
#endif

      for (i = 0; i < nb; ++i)
        {
          /* Update A(i:n,i) */
          i_n = n - i;
          idw = ((offset+i)/nb)%num_gpus;
#define ONGPU
#if defined(ONGPU) && !defined(PRECISION_z) && !defined(PRECISION_c)
          //blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, A(i, 0), &lda, 
          //              W(i, 0), &ldw, &c_one, A(i, i), &ione);
          if( i > 0 ) {
              //int ny = nb;  // always on CPU
              //int ny = -1;  // always on GPU
              int ny = nb0 / 2 ; // half of the time on GPU
              if( i > ny ) {
                   for( id=0; id<num_gpus; id++ ) 
                  {
                      magma_setdevice(id);
                      //magmablasSetKernelStream(stream[id][0]);
                      //cublasSetVector(i_n, sizeof(cuDoubleComplex), A(i, i),   1, dW1(id, i, i),   1);          
                      magma_zsetvector_async( i_n, A(i, i), 1, dW1(id, i, i), 1, stream[id][0] );
                  }
                  int nlocal = (i_n + num_gpus - 1)/num_gpus, i_ni, ii;
                   for( id=0; id<num_gpus; id++ ) 
                  {
                      magma_setdevice(id);
                      magmablasSetKernelStream(stream[id][0]);

                      i_ni = min( nlocal, i_n-id*nlocal );
                      ii   = i + id*nlocal;

                      trace_gpu_start( id, 0, "gemv", "gemv1" );
                      #if defined(PRECISION_z) || defined(PRECISION_c)
                        magmablas_zgemvt('N', i_ni, i, c_neg_one, dW1(id, ii, 0), lddw, 
                                           dW(id, i, 0), lddw, c_one, dW1(id, ii, i), ione);
                        magmablas_zgemvt('N', i_ni, i, c_neg_one, dW(id, ii, 0), lddw, 
                                           dW1(id, i, 0), lddw, c_one, dW1(id, ii, i), ione);
                      #else
                        magma_zgemv(MagmaNoTrans, i_ni, i, c_neg_one, dW1(id, ii, 0), lddw, 
                                      dW(id, i, 0), lddw, c_one, dW1(id, ii, i), ione);
                        magma_zgemv(MagmaNoTrans, i_ni, i, c_neg_one, dW(id, ii, 0), lddw, 
                                      dW1(id, i, 0), lddw, c_one, dW1(id, ii, i), ione);
                      #endif
                      //cublasGetVector(i_n, sizeof(cuDoubleComplex), dW1(id, i, i), 1, A(i, i), 1);          
                      magma_zgetvector_async( i_ni, dW1(id, ii, i), 1, A(ii, i), 1, stream[id][0] );
                      magmablasSetKernelStream(NULL);
                      trace_gpu_end( id, 0 );
                }
              } else {
                  #if defined(PRECISION_z) || defined(PRECISION_c)
                      lapackf77_zlacgv(&i, W(i, 0), &ldw);
                  #endif
                  blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, A(i, 0), &lda, 
                                W(i, 0), &ldw, &c_one, A(i, i), &ione);
                  #if defined(PRECISION_z) || defined(PRECISION_c)
                      lapackf77_zlacgv(&i, W(i, 0), &ldw);
                      lapackf77_zlacgv(&i, A(i ,0), &lda);
                  #endif
                  blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, W(i, 0), &ldw, 
                                A(i, 0), &lda, &c_one, A(i, i), &ione);
                  #if defined(PRECISION_z) || defined(PRECISION_c)
                      lapackf77_zlacgv(&i, A(i, 0), &lda);
                  #endif
              }
          }
          for( id=0; id<num_gpus; id++ ) 
          {
              magma_setdevice(id);
              magma_queue_sync( stream[id][0] );
          }
          magma_setdevice(0);
          //blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, W(i, 0), &ldw, 
          //                A(i, 0), &lda, &c_one, A(i, i), &ione);
#else
          trace_cpu_start( 0, "gemv", "gemv1" );
          if( i > 0 ) {
              #if defined(PRECISION_z) || defined(PRECISION_c)
                  lapackf77_zlacgv(&i, W(i, 0), &ldw);
              #endif
              blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, A(i, 0), &lda, 
                            W(i, 0), &ldw, &c_one, A(i, i), &ione);
              #if defined(PRECISION_z) || defined(PRECISION_c)
                  lapackf77_zlacgv(&i, W(i, 0), &ldw);
                  lapackf77_zlacgv(&i, A(i ,0), &lda);
              #endif
              blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, W(i, 0), &ldw, 
                            A(i, 0), &lda, &c_one, A(i, i), &ione);
              #if defined(PRECISION_z) || defined(PRECISION_c)
                  lapackf77_zlacgv(&i, A(i, 0), &lda);
              #endif
          } else {
            for( id=0; id<num_gpus; id++ ) 
            {
              magma_setdevice(id);
              magma_queue_sync( stream[id][0] );
            }
            magma_setdevice(0);
          }
          trace_cpu_end( 0 );
#endif

          if (i < n-1) 
            {
              /* Generate elementary reflector H(i) to annihilate A(i+2:n,i) */
              i_n = n - i - 1;
              trace_cpu_start( 0, "larfg", "larfg" );
              alpha = *A(i+1, i);
              lapackf77_zlarfg(&i_n, &alpha, A(min(i+2,n-1), i), &ione, &tau[i]);
              e[i] = MAGMA_Z_REAL( alpha );
              MAGMA_Z_SET2REAL(*A(i+1, i), 1.);
              trace_cpu_end( 0 );

              /* Compute W(i+1:n,i) */ 
              // 1. Send the block reflector  A(i+1:n,i) to the GPU
              //trace_gpu_start(  idw, 0, "comm", "comm1" );
              for( id=0; id<num_gpus; id++ ) {
                magma_setdevice(id);
                magma_zsetvector_async( n, A(0, i), 1, dW1(id, 0, i), 1, stream[id][0] );
              }
#ifndef  MAGMABLAS_ZHEMV_MGPU
              magma_setdevice(idw);
              //magmablasSetKernelStream(stream[idw][0]);
              magma_zsetvector( i_n, A(i+1, i), 1, dA(idw, i+1, i), 1 );          
#endif
              //trace_gpu_end( idw, 0 );
              for( id=0; id<num_gpus; id++ ) {
                  magma_setdevice(id);

#ifdef  MAGMABLAS_ZHEMV_MGPU
                  //cublasSetVector(i_n, sizeof(cuDoubleComplex), A(i+1, i), 1, &dx[id][offset+i+1], 1);          
                  magma_zsetvector_async( i_n, A(i+1, i), 1, &dx[id][offset+i+1], 1, stream[id][0] );
#else
                  magma_zsetvector( i_n, A(i+1, i), 1, dx[id], 1 );          
#endif
              }
              /* mat-vec on multiple GPUs */
#ifdef PROFILE_SYMV
              magma_setdevice(0);
              magma_event_record(start, stream[0][0]);
#endif
              magmablas_zhemv_mgpu(num_gpus, k, 'L', i_n, nb0, c_one, da, ldda, offset+i+1, 
                                     dx, ione, c_zero, dy, ione, dwork, ldwork,
                                     work, W(i+1,i), stream );
              //magma_zhemv(MagmaLower, i_n, c_one, dA(0, i+1, i+1), ldda, dA(0, i+1, i), ione, c_zero,
              //            dW(0, i+1, i), ione);
              //cudaMemcpy2DAsync(W(i+1, i), ldw*sizeof(cuDoubleComplex),
              //                  dW(0, i+1, i), lddw*sizeof(cuDoubleComplex),
              //                  sizeof(cuDoubleComplex)*i_n, 1,
              //                  cudaMemcpyDeviceToHost,stream[0][0]);
              //magma_queue_sync( stream[0][0] );
#ifdef PROFILE_SYMV
              magma_setdevice(0);
              magma_event_record(stop, stream[0][0]);
#endif
              trace_cpu_start( 0, "gemv", "gemv2" );
              blasf77_zgemv(MagmaConjTransStr, &i_n, &i, &c_one, W(i+1, 0), &ldw, 
                            A(i+1, i), &ione, &c_zero, W(0, i), &ione);
              blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, A(i+1, 0), &lda, 
                            W(0, i), &ione, &c_zero, f, &ione);
              blasf77_zgemv(MagmaConjTransStr, &i_n, &i, &c_one, A(i+1, 0), &lda, 
                            A(i+1, i), &ione, &c_zero, W(0, i), &ione);
              trace_cpu_end( 0 );

              /* synchronize */
              magmablas_zhemv_synch(num_gpus, k, 'L', i_n, work, W(i+1,i), stream );
#ifdef PROFILE_SYMV
              cudaEventElapsedTime(&etime, start, stop);
              mv_time += (etime/1000.0);
#endif
              trace_cpu_start( 0, "axpy", "axpy" );
              if (i!=0)
                blasf77_zaxpy(&i_n, &c_one, f, &ione, W(i+1, i), &ione);
     
              blasf77_zgemv("No transpose", &i_n, &i, &c_neg_one, W(i+1, 0), &ldw, 
                            W(0, i), &ione, &c_one, W(i+1, i), &ione);
              blasf77_zscal(&i_n, &tau[i], W(i+1,i), &ione);
              
              #if defined(PRECISION_z) || defined(PRECISION_c)
              cblas_zdotc_sub( i_n, W(i+1,i), ione, A(i+1,i), ione, &value );
              #else
              value = cblas_zdotc( i_n, W(i+1,i), ione, A(i+1,i), ione );
              #endif
              alpha = tau[i] * -0.5f * value;
              blasf77_zaxpy(&i_n, &alpha, A(i+1, i), &ione, W(i+1,i), &ione);
              trace_cpu_end( 0 );
              for( id=0; id<num_gpus; id++ ) {
                magma_setdevice(id);
                magma_zsetvector_async( n, W(0, i), 1, dW(id, 0, i), 1, stream[id][0] );
              }
            }
        }
#ifdef PROFILE_SYMV
        magma_setdevice(0);
        magma_event_create( &start );
        magma_event_create( &stop  );
#endif
    }

    magma_free_cpu(f);
    //magma_free_pinned( work );

    return mv_time;
} /* zlatrd_ */

extern "C"
magma_int_t
magmablas_zhemv_mgpu( magma_int_t num_gpus, magma_int_t k, char uplo, 
                      magma_int_t n, magma_int_t nb, 
                      cuDoubleComplex alpha,
                      cuDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
                      cuDoubleComplex **dx, magma_int_t incx,
                      cuDoubleComplex beta,
                      cuDoubleComplex **dy, magma_int_t incy, 
                      cuDoubleComplex **dwork, magma_int_t ldwork,
                      cuDoubleComplex *work, cuDoubleComplex *w,
                      cudaStream_t stream[][10] ) {
#define dX(id, i)    (dx[(id)]+incx*(i))
#define dY(id, i, j) (dy[(id)]+incy*(i)+n*(j))

    cuDoubleComplex c_one = MAGMA_Z_ONE;
    magma_int_t i, ii, j, kk, ib, ib0, id, i_0 = n, i_1, i_local, idw,
                loffset0 = nb*(offset/(nb*num_gpus)), 
                loffset1 = offset%nb, loffset;

    //printf( " n=%d alpha=%e ldda=%d incx=%d beta=%e incy=%d ldwork=%d num_gpus=%d nb=%d offset=%d\n",n,alpha,ldda,incx,beta,incy,ldwork,num_gpus,nb,offset );
#ifdef  MAGMABLAS_ZHEMV_MGPU
    /*if( offset == 64 ) {
        int ii, jj;
        cuDoubleComplex *a;
        magma_setdevice(0);
        magma_zmalloc_pinned( &a, (n + offset)*ldda );
        magma_zgetmatrix( n+offset, n+offset, da[0], ldda, a, ldda );
        for( ii=0; ii<n+offset; ii++ ) {
            for( jj=0; jj<n+offset; jj++ ) printf( "%.16e ",a[ii+jj*ldda] );
            printf( "\n" );
        }
        printf( "\n" );
        magma_zgetmatrix( n+offset, 1, dx[0], ldda, a, ldda );
        for( ii=0; ii<n+offset; ii++ ) printf( "%.16e\n",a[ii] );
        printf( "\n" );
        magma_free_pinned( a );
    }*/
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        trace_gpu_start( id, 0, "symv", "symv" );
        cudaMemset( dwork[id], 0, ldwork*sizeof(cuDoubleComplex) );
    }

    if( nb == 32 ) {
      magmablas_zhemv_mgpu_32_offset( 'L', offset+n, alpha, da, ldda, 
                                         dx, incx, 
                                      beta, 
                                      dy, incy, 
                                      dwork, ldwork,
                                      num_gpus, nb, offset,
                                      stream );
    } else {
      magmablas_zhemv_mgpu_offset( 'L', offset+n, alpha, da, ldda, 
                                   dx, incx, 
                                   beta, 
                                   dy, incy, 
                                   dwork, ldwork,
                                   num_gpus, nb, offset,
                                   stream );
    }
    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
    //magma_setdevice(0);
    //magmablasSetKernelStream(stream[0][0]);
    //magma_zhemv(MagmaLower, n, alpha, &da[0][offset+offset*ldda], ldda, &dx[0][offset], incx, beta, &dy[0][offset], incy );
    //magmablasSetKernelStream(NULL);

    /*if( offset == 64 ) {
        int ii, jj;
        cuDoubleComplex *a;
        magma_setdevice(0);
        magma_zmalloc_pinned( &a, (offset + n)*ldda );
        printf( ">>>\n" );
        magma_zgetmatrix( (offset+n), 1, dy[0], ldda, a, ldda );
        //for( ii=0; ii<offset+n; ii++ ) printf( "%d: %e\n",ii,a[ii] );
        for( ii=offset; ii<offset+n; ii++ ) printf( "%.16e\n",ii,a[ii] );
        printf( ">>>\n" );
        printf( "\n" );
        magma_free_pinned( a );
    }*/

    /* send to CPU */
    magma_setdevice(0);
    //trace_gpu_start(  0, 0, "comm", "comm2" );
    magma_zgetvector_async( n, dY(0, offset, 0), 1, w, 1, stream[0][0] );
    //trace_gpu_end( 0, 0 );
    magmablasSetKernelStream(NULL);

    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        //trace_gpu_start(  id, 0, "comm", "comm3" );
        magma_zgetvector_async( n, dY(id, offset, 0), 1, &work[id*n], 1, stream[id][0] );
        //trace_gpu_end( id, 0 );
        magmablasSetKernelStream(NULL);
    }
#else
    //magma_zhemv(uplo, n, alpha, da, ldda, dx, incx, beta, dy, incy );
    
    idw = (offset/nb)%num_gpus;

    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        cudaMemset( dy[id], 0, n*k*sizeof(cuDoubleComplex) );
    }

    /* the first block */
    if( loffset1 > 0 ) {
      id = idw;
      kk = 0;
      //printf( " ** i=0 (%d,%d) **\n",id,kk );

      magma_setdevice(id);
      magmablasSetKernelStream(stream[id][kk]);

      loffset = loffset0+loffset1;
      ib0 = min(nb-loffset1,n);
      // diagonal
      magma_zhemv(MagmaLower, ib0, c_one, dA(id, 0, 0 ), ldda,
                  dX(id, 0), incx, c_one, dY(id, 0, kk), incy);
      // off-diagonl
      if( ib0 < n ) {
        //i_1 = n - ib0;
        for( j=ib0; j<n; j+= i_0 ) {
          i_1 = min(i_0, n-j);
          magma_zgemv(MagmaNoTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                      dX(id, 0), incx, c_one, dY(id, j, kk), incy);
          magma_zgemv(MagmaConjTrans, i_1, ib0, c_one, dA(id, j, 0), ldda,
                      dX(id, j), incx, c_one, dY(id, 0, kk), incy);
        }
      }
    } else {
      ib0 = 0;
    }

    /* diagonal */
    for( i=ib0; i<n; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = ((i+loffset1)/(nb*num_gpus))%k;
        //printf( " ** i=%d (%d,%d) **\n",i,id,kk );

        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        i_local = (i+loffset1)/(nb*num_gpus);
        ib = min(nb,n-i);

        ii = nb*i_local;

        loffset = loffset0;
        if( id < idw ) loffset += nb;
        magma_zhemv(MagmaLower,  ib, c_one, dA(id, i, ii), ldda,
                    dX(id, i), incx, c_one, dY(id, i, kk), incy);
    }

    /* off-diagonal */
    for( i=ib0; i<n-nb; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = ((i+loffset1)/(nb*num_gpus))%k;
        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][kk]);
        //printf( " ** i=%d (%d,%d) **\n",i,id,kk );

        i_local = ((i+loffset1)/nb)/num_gpus;
        ii = nb*i_local;
        ib = min(nb,n-i);
        loffset = loffset0;
        if( id < idw ) loffset += nb;

        //i_0 = n - (i+ib);
        for( j=i+ib; j<n; j+= i_0 ) {
          i_1 = min(i_0, n-j);
          magma_zgemv(MagmaNoTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                      dX(id, i), incx, c_one, dY(id, j, kk), incy);
          magma_zgemv(MagmaConjTrans, i_1, ib, c_one, dA(id, j, ii), ldda,
                      dX(id, j), incx, c_one, dY(id, i, kk), incy);
        }
    }

    /* send to CPU */
    magma_setdevice(0);
    magma_zgetvector_async( n, dY(0, 0, 0), 1, w, 1, stream[0][0] );
    for( kk=1; kk<k; kk++ ) 
      magma_zgetvector_async( n, dY(0, 0, kk), 1, &work[kk*n], 1, stream[0][kk] );
    magmablasSetKernelStream(NULL);

    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk<k; kk++ ) {
          magma_zgetvector_async( n, dY(id, 0, kk), 1, &work[id*k*n + kk*n], 1, stream[id][kk] );
        }
        magmablasSetKernelStream(NULL);
    }
#endif
    return 0;
}

extern "C"
magma_int_t
magmablas_zhemv_synch( magma_int_t num_gpus, magma_int_t k, char uplo, 
                      magma_int_t n, cuDoubleComplex *work, cuDoubleComplex *w,
                      cudaStream_t stream[][10] ) {

    cuDoubleComplex c_one = MAGMA_Z_ONE;
    magma_int_t id, ione = 1, kk;

    /* reduce on CPU */
    magma_setdevice(0);
    magma_queue_sync( stream[0][0] );
    for( kk=1; kk<k; kk++ ) {
        magma_queue_sync( stream[0][kk] );
        blasf77_zaxpy( &n, &c_one, &work[kk*n], &ione, w, &ione );
    }
    //for( kk=0; kk<n; kk++ ) printf( " work[%d]=%e\n",kk,w[kk] );
    for( id=1; id<num_gpus; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk<k; kk++ ) {
            magma_queue_sync( stream[id][kk] );
            blasf77_zaxpy( &n, &c_one, &work[id*k*n + kk*n], &ione, w, &ione );
        }
    }

    return 0;
}
