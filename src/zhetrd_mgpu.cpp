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

#if (GPUSHMEM >= 200)

extern "C" magma_int_t
magma_zhtodhe(int num_gpus, char *uplo, magma_int_t n, magma_int_t nb,
              cuDoubleComplex *a, magma_int_t lda, cuDoubleComplex **dwork, magma_int_t ldda, cudaStream_t stream[][10],
              magma_int_t *info);

extern "C" void
magma_zher2k_mgpu(int num_gpus, char uplo, char trans, int nb, int n, int k,
        cuDoubleComplex alpha, cuDoubleComplex **db, int lddb, int offset_b,
        double beta,           cuDoubleComplex **dc, int lddc, int offset,
        int num_streams, cudaStream_t stream[][10]);

extern "C" double     
magma_zlatrd_mgpu(int num_gpus, char uplo,  
                  magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
                  cuDoubleComplex *a,  magma_int_t lda, 
                  double *e, cuDoubleComplex *tau, 
                  cuDoubleComplex *w,   magma_int_t ldw,
                  cuDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
                  cuDoubleComplex **dw, magma_int_t lddw,
                  cuDoubleComplex *dwork[4], magma_int_t ldwork,
                  magma_int_t k,
                  cuDoubleComplex  *dx[4], cuDoubleComplex *dy[4],
                  cuDoubleComplex *work,
                  cudaStream_t stream[][10],
                  double *times);


// === Define what BLAS to use ============================================
#define PRECISION_z

#if (defined(PRECISION_s))
//  #define cublasSsyr2k magmablas_ssyr2k
#endif
// === End defining what BLAS to use ======================================

#define  A(i, j) ( a+(j)*lda  + (i))
#define dA(id, i, j) (da[(id)]+(j)*ldda + (i))
#define dW(id, i, j) (dwork[(id)]+(j)*ldda + (i))

extern "C" magma_int_t
magma_zhetrd_mgpu(int num_gpus, int k, char uplo, magma_int_t n, 
             cuDoubleComplex *a, magma_int_t lda, 
             double *d, double *e, cuDoubleComplex *tau,
             cuDoubleComplex *work, magma_int_t lwork, 
             magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   
    ZHETRD reduces a complex Hermitian matrix A to real symmetric   
    tridiagonal form T by an orthogonal similarity transformation:   
    Q\*\*H * A * Q = T.   

    Arguments   
    =========   
    UPLO    (input) CHARACTER*1   
            = 'U':  Upper triangle of A is stored;   
            = 'L':  Lower triangle of A is stored.   

    N       (input) INTEGER   
            The order of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)   
            On entry, the Hermitian matrix A.  If UPLO = 'U', the leading   
            N-by-N upper triangular part of A contains the upper   
            triangular part of the matrix A, and the strictly lower   
            triangular part of A is not referenced.  If UPLO = 'L', the   
            leading N-by-N lower triangular part of A contains the lower   
            triangular part of the matrix A, and the strictly upper   
            triangular part of A is not referenced.   
            On exit, if UPLO = 'U', the diagonal and first superdiagonal   
            of A are overwritten by the corresponding elements of the   
            tridiagonal matrix T, and the elements above the first   
            superdiagonal, with the array TAU, represent the orthogonal   
            matrix Q as a product of elementary reflectors; if UPLO   
            = 'L', the diagonal and first subdiagonal of A are over-   
            written by the corresponding elements of the tridiagonal   
            matrix T, and the elements below the first subdiagonal, with   
            the array TAU, represent the orthogonal matrix Q as a product   
            of elementary reflectors. See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    D       (output) COMPLEX_16 array, dimension (N)   
            The diagonal elements of the tridiagonal matrix T:   
            D(i) = A(i,i).   

    E       (output) COMPLEX_16 array, dimension (N-1)   
            The off-diagonal elements of the tridiagonal matrix T:   
            E(i) = A(i,i+1) if UPLO = 'U', E(i) = A(i+1,i) if UPLO = 'L'.   

    TAU     (output) COMPLEX_16 array, dimension (N-1)   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    WORK    (workspace/output) COMPLEX_16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO = 0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK.  LWORK >= 1.   
            For optimum performance LWORK >= N*NB, where NB is the   
            optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0:  successful exit   
            < 0:  if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    If UPLO = 'U', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(n-1) . . . H(2) H(1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with   
    v(i+1:n) = 0 and v(i) = 1; v(1:i-1) is stored on exit in   
    A(1:i-1,i+1), and tau in TAU(i).   

    If UPLO = 'L', the matrix Q is represented as a product of elementary   
    reflectors   

       Q = H(1) H(2) . . . H(n-1).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i) = 0 and v(i+1) = 1; v(i+2:n) is stored on exit in A(i+2:n,i),   
    and tau in TAU(i).

    The contents of A on exit are illustrated by the following examples   
    with n = 5:   

    if UPLO = 'U':                       if UPLO = 'L':   

      (  d   e   v2  v3  v4 )              (  d                  )   
      (      d   e   v3  v4 )              (  e   d              )   
      (          d   e   v4 )              (  v1  e   d          )   
      (              d   e  )              (  v1  v2  e   d      )   
      (                  d  )              (  v1  v2  v3  e   d  )   

    where d and e denote diagonal and off-diagonal elements of T, and vi   
    denotes an element of the vector defining H(i).   
    =====================================================================    */  

    char uplo_[2] = {uplo, 0};

    magma_int_t ln, ldda;
    magma_int_t nb = magma_get_zhetrd_nb(n), ib; 

    cuDoubleComplex z_neg_one = MAGMA_Z_NEG_ONE;
    cuDoubleComplex z_one = MAGMA_Z_ONE;
    double  d_one = MAGMA_D_ONE;
    double mv_time = 0.0;
    double up_time = 0.0;
    
    static magma_int_t kk, nx;
    static magma_int_t i = 0, ii, iii, j, did, i_n;
    static magma_int_t iinfo;
    static magma_int_t ldwork, lddwork, lwkopt, ldwork2;
    static magma_int_t lquery;
    cudaStream_t stream[4][10];
    cuDoubleComplex *dx[4], *dy[4], *hwork;
    cuDoubleComplex *dwork2[4]; 

    *info = 0;
    long int upper = lapackf77_lsame(uplo_, "U");
    lquery = lwork == -1;
    if ( !upper && !lapackf77_lsame(uplo_, "L")) {
        printf( " uplo = %c\n",uplo );
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    } else if (lwork < nb*n && ! lquery) {
        *info = -9;
    } else if( k > 2 ) {
    *info = 2;
    }

    if (*info == 0) {
      /* Determine the block size. */
      ldwork = lddwork = n;
      lwkopt = n * nb;
      MAGMA_Z_SET2REAL( work[0], lwkopt );
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return MAGMA_ERR_ILLEGAL_VALUE;
    }
    else if (lquery)
      return 0;

    /* Quick return if possible */
    if (n == 0) {
        work[0] = z_one;
        return 0;
    }

    cuDoubleComplex *da[4];
    cuDoubleComplex *dwork[4]; 

    double times[11];
    for( did=0; did<11; did++ ) times[did] = 0;
//#define PROFILE_SY2RK
#ifdef  PROFILE_SY2RK
    cudaEvent_t start, stop;
    float etime;
    cudaSetDevice(0);
    cudaEventCreate( &start );
    cudaEventCreate( &stop  );
#endif
    /* need to be multiple of 128 and set to be zeros for gemvt? */
    ldda = lda;
    ln = ((nb*(1+n/(nb*num_gpus))+31)/32)*32;
    ldwork2 = (1+ n / nb + (n % nb != 0)) * ldda;
    for( did=0; did<num_gpus; did++ ) {
      cudaSetDevice(did);
      if ( MAGMA_SUCCESS != cublasAlloc(ln*ldda+3*lddwork*nb, sizeof(cuDoubleComplex), (void**)&da[did]) ||
           MAGMA_SUCCESS != cublasAlloc(k*n,     sizeof(cuDoubleComplex), (void**)&dx[did]) ||
           MAGMA_SUCCESS != cublasAlloc(k*n,     sizeof(cuDoubleComplex), (void**)&dy[did]) ||
           MAGMA_SUCCESS != cublasAlloc(ldwork2, sizeof(cuDoubleComplex), (void**)&dwork2[did] ) ) {
        for( i=0; i<did; i++ ) {
            cudaSetDevice(i);
            cublasFree(da[i]);
            cublasFree(dx[i]);
            cublasFree(dy[i]);
        }
        *info = MAGMA_ERR_DEVICE_ALLOC;
        return *info;
      }
      dwork[did] = da[did] + ln*ldda;

      for( kk=0; kk<k; kk++ ) cudaStreamCreate(&stream[did][kk]);
    }
    if( MAGMA_SUCCESS != cudaMallocHost( (void**)&hwork, k*num_gpus*n*sizeof(cuDoubleComplex) ) ) {
      for( i=0; i<num_gpus; i++ ) {
        cudaSetDevice(i);
        cublasFree(da[i]);
        cublasFree(dx[i]);
        cublasFree(dy[i]);
      }
      *info = MAGMA_ERR_HOST_ALLOC;
      return *info;
    }

    if (n < 2048)
      nx = n;
    else
      nx = 512;

    if (upper) {

        /* Copy the matrix to the GPU */ 
        if (1<=n-nx) {
          magma_zhtodhe(num_gpus, &uplo, n, nb, a, lda, da, ldda, stream, &iinfo );
        }

        /*  Reduce the upper triangle of A.   
            Columns 1:kk are handled by the unblocked method. */
        for (i = nb*((n-1)/nb); i >= nx; i -= nb) 
          {
            ib = min(nb, n-i);

            ii  = nb*(i/(nb*num_gpus));
            did = (i/nb)%num_gpus;

            /* Reduce columns i:i+nb-1 to tridiagonal form and form the   
               matrix W which is needed to update the unreduced part of   
               the matrix */
            
            /*   Get the current panel (no need for the 1st iteration) */
            if (i!=nb*((n-1)/nb)) {
              cudaSetDevice(did);
              cudaMemcpy2DAsync(  A(0, i),       lda *sizeof(cuDoubleComplex),
                                 dA(did, 0, ii), ldda*sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*(i+ib), ib,
                                  cudaMemcpyDeviceToHost,stream[did][0]);
            }

            magma_zlatrd_mgpu(num_gpus, uplo, n, i+ib, ib, nb, 
                              A(0, 0), lda, e, tau, 
                              work, ldwork, 
                              da, ldda, 0, 
                              dwork, i+ib, 
                              dwork2, ldwork2, 
                              1, dx, dy, hwork, 
                              stream, times);

            magma_zher2k_mgpu(num_gpus, 'U', 'N', nb, i, ib, 
                         z_neg_one, dwork, i+ib, 0,
                         d_one,     da,    ldda, 0, 
                         k, stream);

            /* Copy superdiagonal elements back into A, and diagonal   
               elements into D */
            for (j = i; j < i+ib; ++j) {
                if( j > 0 ) { MAGMA_Z_SET2REAL( *A(j-1, j), e[j - 1] ); }
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        } /* end of for i=... */
      
        if( nx > 0 ) {
          if (1<=n-nx) { /* else A is already on CPU */
            did = 0;
            cudaSetDevice(did);
            cublasGetMatrix(nx, nx, sizeof(cuDoubleComplex), dA(did, 0, 0), ldda,
                            A(0, 0), lda);

            int iii = i;
            for (i=0; i < nx; i += nb) {
                ib = min(nb, n-i);
                ii  = nb*(i/(nb*num_gpus));
                did = (i/nb)%num_gpus;

                cudaSetDevice(did);
                cudaMemcpy2DAsync( A(0, i),        lda *sizeof(cuDoubleComplex),
                                   dA(did, 0, ii), ldda*sizeof(cuDoubleComplex),
                                   sizeof(cuDoubleComplex)*nx, ib,
                                   cudaMemcpyDeviceToHost,stream[did][0]);
            }
            for( did=0; did<num_gpus; did++ ) {
                cudaSetDevice(did);
                cudaStreamSynchronize(stream[did][0]);
            }
          }

          for( did=0; did<num_gpus; did++ ) {
                cudaSetDevice(did);
                cudaStreamSynchronize(stream[did][0]);
          }
          /*  Use unblocked code to reduce the last or only block */
          lapackf77_zhetd2(uplo_, &nx, A(0, 0), &lda, d, e, tau, &iinfo);
        }
    } else {
        trace_init( 1, num_gpus, k, (CUstream_st**)stream );
        /* Copy the matrix to the GPU */
        if (1<=n-nx) {
          magma_zhtodhe(num_gpus, &uplo, n, nb, a, lda, da, ldda, stream, &iinfo );
        }

        /* Reduce the lower triangle of A */
        for (i = 0; i < n-nx; i += nb) {
            ib = min(nb, n-i);

            ii  = nb*(i/(nb*num_gpus));
            did = (i/nb)%num_gpus;
            /* Reduce columns i:i+ib-1 to tridiagonal form and form the
               matrix W which is needed to update the unreduced part of
               the matrix */

            /*   Get the current panel (no need for the 1st iteration) */
            if (i!=0) {
              cudaSetDevice(did);
              trace_gpu_start( did, 0, "comm", "get" );
              cudaMemcpy2DAsync(  A(i, i),       lda *sizeof(cuDoubleComplex),
                                 dA(did, i, ii), ldda*sizeof(cuDoubleComplex),
                                  sizeof(cuDoubleComplex)*(n-i), ib,
                                  cudaMemcpyDeviceToHost,stream[did][0]);
              trace_gpu_end( did, 0 );
              cudaSetDevice(0);
            }
            
            
            mv_time += 
            magma_zlatrd_mgpu(num_gpus, uplo, n, n-i, ib, nb,
                              A(i, i), lda, &e[i], 
                              &tau[i], work, ldwork, 
                              da, ldda, i, 
                              dwork,  (n-i),
                              dwork2, ldwork2,
                              1, dx, dy, hwork,
                              stream, times );
                              
#ifdef PROFILE_SY2RK
            cudaSetDevice(0);
            if( i > 0 ) {
              cudaEventElapsedTime(&etime, start, stop);
              up_time += (etime/1000.0);
            } 
            cudaEventRecord(start, 0);
#endif
            magma_zher2k_mgpu(num_gpus, 'L', 'N', nb, n-i-ib, ib, 
                         z_neg_one, dwork, n-i, ib,
                         d_one, da, ldda, i+ib, k, stream);
#ifdef PROFILE_SY2RK
            cudaSetDevice(0);
            cudaEventRecord(stop, 0);
#endif

            /* Copy subdiagonal elements back into A, and diagonal   
               elements into D */
            for (j = i; j < i+ib; ++j) {
                if( j+1 < n ) { MAGMA_Z_SET2REAL( *A(j+1, j), e[j] ); }
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }
        } /* for i=... */

        /* Use unblocked code to reduce the last or only block */
        if ( i < n ) {
          int iii = i;
          i_n = n-i;
          if( i > 0 ) {
            for (; i < n; i += nb) {
                ib = min(nb, n-i);
                ii  = nb*(i/(nb*num_gpus));
                did = (i/nb)%num_gpus;

                cudaSetDevice(did);
                cudaMemcpy2DAsync( A(iii, i),       lda *sizeof(cuDoubleComplex),
                                   dA(did, iii, ii), ldda*sizeof(cuDoubleComplex),
                                    sizeof(cuDoubleComplex)*i_n, ib,
                                  cudaMemcpyDeviceToHost,stream[did][0]);
            }
            for( did=0; did<num_gpus; did++ ) {
                cudaSetDevice(did);
                cudaStreamSynchronize(stream[did][0]);
            }
          }
          lapackf77_zhetrd(uplo_, &i_n, A(iii, iii), &lda, &d[iii], &e[iii],
                           &tau[iii], work, &lwork, &iinfo);
        }
      }
#ifdef PROFILE_SY2RK
   cudaSetDevice(0);
   if( n > nx ) {
        cudaEventElapsedTime(&etime, start, stop);
        up_time += (etime/1000.0);
    } 
    cudaEventDestroy( start );
    cudaEventDestroy( stop  );
#endif
 
    trace_finalize( "zhetrd.svg","trace.css" );
    for( did=0; did<num_gpus; did++ ) {
      cudaSetDevice(did);
      for( kk=0; kk<k; kk++ ) cudaStreamSynchronize(stream[did][kk]);
      for( kk=0; kk<k; kk++ ) cudaStreamDestroy(stream[did][kk]);
      cublasFree(da[did]);
      cublasFree(dx[did]);
      cublasFree(dy[did]);
      cublasFree(dwork2[did]);
    }
    cudaSetDevice(0);
    cudaFreeHost(hwork);
    MAGMA_Z_SET2REAL( work[0], lwkopt );

#ifdef PROFILE_SY2RK
    printf( " n=%d nb=%d\n",n,nb );
    printf( " Time in ZLARFG: %.2e seconds\n",times[0] );
    printf( " Time in ZHEMV : %.2e seconds\n",mv_time );
    printf( " Time in ZHER2K: %.2e seconds\n",up_time );
#endif
    return MAGMA_SUCCESS;
} /* magma_zhetrd */

extern "C" magma_int_t
magma_zhtodhe(int num_gpus, char *uplo, magma_int_t n, magma_int_t nb,
              cuDoubleComplex *a, magma_int_t lda, cuDoubleComplex **da, magma_int_t ldda, cudaStream_t stream[][10],
              magma_int_t *info) {

      magma_int_t k;
      if( lapackf77_lsame(uplo, "L") ) {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j<n; j+=nb) {
          jj =  j/(nb*num_gpus);
          k  = (j/nb)%num_gpus;

          jb = min(nb, (n-j));
          mj = n-j;

          cudaSetDevice(k);
          cudaMemcpy2DAsync( dA(k, j, jj*nb), ldda*sizeof(cuDoubleComplex),
                             A(j, j),         lda *sizeof(cuDoubleComplex),
                             sizeof(cuDoubleComplex)*mj, jb,
                             cudaMemcpyHostToDevice, stream[k][0]);
        }
      } else {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j<n; j+=nb) {
          jj =  j/(nb*num_gpus);
          k  = (j/nb)%num_gpus;

          jb = min(nb, (n-j));
          //mj = n-j;
          mj = j+jb;

          cudaSetDevice(k);
          cudaMemcpy2DAsync( dA(k, 0, jj*nb), ldda*sizeof(cuDoubleComplex),
                             A(0, j),         lda *sizeof(cuDoubleComplex),
                             sizeof(cuDoubleComplex)*mj, jb,
                             cudaMemcpyHostToDevice, stream[k][0]);
        }
      }
      for( k=0; k<num_gpus; k++ ) {
        cudaSetDevice(k);
        cudaStreamSynchronize(stream[k][0]);
      }
      cudaSetDevice(0);

      return MAGMA_SUCCESS;
}

extern "C" void
magma_zher2k_mgpu(int num_gpus, char uplo, char trans, int nb, int n, int k,
    cuDoubleComplex alpha, cuDoubleComplex **db, int lddb, int offset_b,
    double beta,           cuDoubleComplex **dc, int lddc, int offset,
    int num_streams, cudaStream_t stream[][10]) {

#define dB(id, i, j)  (db[(id)]+(j)*lddb + (i)+offset_b)
#define dB1(id, i, j) (db[(id)]+(j)*lddb + (i)+offset_b)+k*lddb
#define dC(id, i, j)  (dc[(id)]+(j)*lddc + (i))

    char uplo_[2]  = {uplo, 0};
    int i, id, ib, ii, kk, n1, m1;
    cuDoubleComplex z_one = MAGMA_Z_ONE;

    /* diagonal update */
    for( i=0; i<n; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        cudaSetDevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));

        /* zher2k on diagonal block */
        trace_gpu_start( id, kk, "syr2k", "syr2k" );
        cublasZher2k(uplo, trans, ib, k, 
              alpha, dB1(id, i,        0 ), lddb, 
                     dB(id,  i,        0 ), lddb,
              beta,  dC(id,  i+offset,   ii), lddc);
        trace_gpu_end( id, kk );
    }

    /* off-diagonal update */
    if( lapackf77_lsame( uplo_, "L" ) ) {
      for( i=0; i<n-nb; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        cudaSetDevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));
        n1 = n-i-ib;

        // zgemm on off-diagonal blocks 
        trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
        cublasZgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k, 
                    alpha, dB1(id, i+ib,        0 ), lddb,
                           dB(id,  i,           0 ), lddb, 
                    z_one, dC(id,  i+offset+ib, ii), lddc);
        trace_gpu_end( id, kk );
      }
    } else {
      for( i=nb; i<n; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        cudaSetDevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));
        cublasZgemm(MagmaNoTrans, MagmaConjTrans, i, ib, k, 
                    alpha, dB1(id, 0, 0 ), lddb,
                           dB(id,  i, 0 ), lddb, 
                    z_one, dC(id,  0, ii), lddc);
      }
    }

    if( lapackf77_lsame( uplo_, "L" ) ) {
      for( i=0; i<n-nb; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        cudaSetDevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));
        n1 = n-i-ib;

        /* zgemm on off-diagonal blocks */
        trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
        cublasZgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k, 
                    alpha, dB(id,  i+ib,        0 ), lddb,
                           dB1(id, i,           0 ), lddb, 
                    z_one, dC(id,  i+offset+ib, ii), lddc);
        trace_gpu_end( id, kk );
      }
    } else {
      for( i=nb; i<n; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        cudaSetDevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));
        cublasZgemm(MagmaNoTrans, MagmaConjTrans, i, ib, k, 
                    alpha, dB( id, 0, 0 ), lddb,
                           dB1(id, i, 0 ), lddb, 
                    z_one, dC(id,  0, ii), lddc);
      }
    }

    for( id=0; id<num_gpus; id++ ) {
        cudaSetDevice(id);
        for( kk=0; kk<num_streams; kk++ ) cudaStreamSynchronize(stream[id][kk]);
        magmablasSetKernelStream(NULL);
    }
}

#endif
