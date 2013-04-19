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
              magmaDoubleComplex *a, magma_int_t lda,
              magmaDoubleComplex **dwork, magma_int_t ldda,
              magma_queue_t stream[][10], magma_int_t *info);

extern "C" void
magma_zher2k_mgpu(int num_gpus, char uplo, char trans, int nb, int n, int k,
        magmaDoubleComplex alpha, magmaDoubleComplex **db, int lddb, int offset_b,
        double beta,           magmaDoubleComplex **dc, int lddc, int offset,
        int num_streams, magma_queue_t stream[][10]);

extern "C" double
magma_zlatrd_mgpu(int num_gpus, char uplo,
                  magma_int_t n0, magma_int_t n, magma_int_t nb, magma_int_t nb0,
                  magmaDoubleComplex *a,  magma_int_t lda,
                  double *e, magmaDoubleComplex *tau,
                  magmaDoubleComplex *w,   magma_int_t ldw,
                  magmaDoubleComplex **da, magma_int_t ldda, magma_int_t offset,
                  magmaDoubleComplex **dw, magma_int_t lddw,
                  magmaDoubleComplex *dwork[MagmaMaxGPUs], magma_int_t ldwork,
                  magma_int_t k,
                  magmaDoubleComplex  *dx[MagmaMaxGPUs], magmaDoubleComplex *dy[MagmaMaxGPUs],
                  magmaDoubleComplex *work,
                  magma_queue_t stream[][10],
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
             magmaDoubleComplex *a, magma_int_t lda,
             double *d, double *e, magmaDoubleComplex *tau,
             magmaDoubleComplex *work, magma_int_t lwork,
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

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    double  d_one = MAGMA_D_ONE;
    double mv_time = 0.0;
    double up_time = 0.0;
    
    magma_int_t kk, nx;
    magma_int_t i = 0, ii, iii, j, did, i_n;
    magma_int_t iinfo;
    magma_int_t ldwork, lddwork, lwkopt, ldwork2;
    magma_int_t lquery;
    magma_queue_t stream[MagmaMaxGPUs][10];
    magmaDoubleComplex *dx[MagmaMaxGPUs], *dy[MagmaMaxGPUs], *hwork;
    magmaDoubleComplex *dwork2[MagmaMaxGPUs];

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
        work[0] = c_one;
        return 0;
    }

    magmaDoubleComplex *da[MagmaMaxGPUs];
    magmaDoubleComplex *dwork[MagmaMaxGPUs];

    double times[11];
    for( did=0; did<11; did++ )
        times[did] = 0;
//#define PROFILE_SY2RK
#ifdef PROFILE_SY2RK
    magma_event_t start, stop;
    float etime;
    magma_setdevice(0);
    magma_event_create( &start );
    magma_event_create( &stop  );
#endif
    ldda = lda;
    ln = ((nb*(1+n/(nb*num_gpus))+31)/32)*32;
    ldwork2 = (1+ n / nb + (n % nb != 0)) * ldda;
    for( did=0; did<num_gpus; did++ ) {
        magma_setdevice(did);
        if ( MAGMA_SUCCESS != magma_zmalloc(&da[did],     ln*ldda+3*lddwork*nb) ||
             MAGMA_SUCCESS != magma_zmalloc(&dx[did],     k*n     ) ||
             MAGMA_SUCCESS != magma_zmalloc(&dy[did],     k*n     ) ||
             MAGMA_SUCCESS != magma_zmalloc(&dwork2[did], ldwork2 ) ) {
            for( i=0; i<did; i++ ) {
                magma_setdevice(i);
                magma_free(da[i]);
                magma_free(dx[i]);
                magma_free(dy[i]);
            }
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        dwork[did] = da[did] + ln*ldda;
        
        for( kk=0; kk<k; kk++ )
            magma_queue_create(&stream[did][kk]);
    }
    magma_setdevice(0);
    if( MAGMA_SUCCESS != magma_zmalloc_pinned( &hwork, k*num_gpus*n ) ) {
        for( i=0; i<num_gpus; i++ ) {
            magma_setdevice(i);
            magma_free(da[i]);
            magma_free(dx[i]);
            magma_free(dy[i]);
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
        for (i = nb*((n-1)/nb); i >= nx; i -= nb) {
            ib = min(nb, n-i);

            ii  = nb*(i/(nb*num_gpus));
            did = (i/nb)%num_gpus;

            /* wait for the next panel */
            if (i!=nb*((n-1)/nb)) {
                magma_setdevice(did);
                magma_queue_sync(stream[did][0]);
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
                         c_neg_one, dwork, i+ib, 0,
                         d_one,     da,    ldda, 0,
                         k, stream);

            /* get the next panel */
            if (i-nb >= nx ) {
                ib = min(nb, n-(i-nb));
                
                ii  = nb*((i-nb)/(nb*num_gpus));
                did = ((i-nb)/nb)%num_gpus;
                magma_setdevice(did);
                
                magma_zgetmatrix_async( (i-nb)+ib, ib,
                                        dA(did, 0, ii), ldda,
                                         A(0, i-nb),    lda,
                                        stream[did][0] );
            }

            /* Copy superdiagonal elements back into A, and diagonal
               elements into D */
            for (j = i; j < i+ib; ++j) {
                if( j > 0 ) { MAGMA_Z_SET2REAL( *A(j-1, j), e[j - 1] ); }
                d[j] = MAGMA_Z_REAL( *A(j, j) );
            }

        } /* end of for i=... */
      
        if( nx > 0 ) {
            if (1<=n-nx) { /* else A is already on CPU */
                int iii = i;
                for (i=0; i < nx; i += nb) {
                    ib = min(nb, n-i);
                    ii  = nb*(i/(nb*num_gpus));
                    did = (i/nb)%num_gpus;
                
                    magma_setdevice(did);
                    magma_zgetmatrix_async( nx, ib,
                                            dA(did, 0, ii), ldda,
                                            A(0, i),        lda,
                                            stream[did][0] );
                }
            }
            
            for( did=0; did<num_gpus; did++ ) {
                magma_setdevice(did);
                magma_queue_sync(stream[did][0]);
            }
            /*  Use unblocked code to reduce the last or only block */
            lapackf77_zhetd2(uplo_, &nx, A(0, 0), &lda, d, e, tau, &iinfo);
        }
    }
    else {
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
                magma_setdevice(did);
                trace_gpu_start( did, 0, "comm", "get" );
                magma_zgetmatrix_async( n-i, ib,
                                        dA(did, i, ii), ldda,
                                         A(i,i),        lda,
                                        stream[did][0] );
                trace_gpu_end( did, 0 );
                magma_queue_sync(stream[did][0]);
                magma_setdevice(0);
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
            magma_setdevice(0);
            if( i > 0 ) {
                cudaEventElapsedTime(&etime, start, stop);
                up_time += (etime/1000.0);
            }
            magma_event_record(start, 0);
#endif
            magma_zher2k_mgpu(num_gpus, 'L', 'N', nb, n-i-ib, ib,
                         c_neg_one, dwork, n-i, ib,
                         d_one, da, ldda, i+ib, k, stream);
#ifdef PROFILE_SY2RK
            magma_setdevice(0);
            magma_event_record(stop, 0);
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
                
                    magma_setdevice(did);
                    magma_zgetmatrix_async( i_n, ib,
                                            dA(did, iii, ii), ldda,
                                             A(iii, i),       lda,
                                            stream[did][0] );
                }
                for( did=0; did<num_gpus; did++ ) {
                    magma_setdevice(did);
                    magma_queue_sync(stream[did][0]);
                }
            }
            lapackf77_zhetrd(uplo_, &i_n, A(iii, iii), &lda, &d[iii], &e[iii],
                             &tau[iii], work, &lwork, &iinfo);
        }
    }
#ifdef PROFILE_SY2RK
    magma_setdevice(0);
    if( n > nx ) {
        cudaEventElapsedTime(&etime, start, stop);
        up_time += (etime/1000.0);
    }
    magma_event_destroy( start );
    magma_event_destroy( stop  );
#endif

    trace_finalize( "zhetrd.svg","trace.css" );
    for( did=0; did<num_gpus; did++ ) {
        magma_setdevice(did);
        for( kk=0; kk<k; kk++ )
            magma_queue_sync(stream[did][kk]);
        for( kk=0; kk<k; kk++ )
            magma_queue_destroy(stream[did][kk]);
        magma_free(da[did]);
        magma_free(dx[did]);
        magma_free(dy[did]);
        magma_free(dwork2[did]);
    }
    magma_setdevice(0);
    magma_free_pinned(hwork);
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
              magmaDoubleComplex *a, magma_int_t lda,
              magmaDoubleComplex **da, magma_int_t ldda,
              magma_queue_t stream[][10], magma_int_t *info)
{
    magma_int_t k;
    if( lapackf77_lsame(uplo, "L") ) {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j<n; j+=nb) {
            jj =  j/(nb*num_gpus);
            k  = (j/nb)%num_gpus;
            
            jb = min(nb, (n-j));
            mj = n-j;
            
            magma_setdevice(k);
            magma_zsetmatrix_async( mj, jb,
                                     A(j,j),         lda,
                                    dA(k, j, jj*nb), ldda,
                                    stream[k][0] );
        }
    }
    else {
        /* go through each block-column */
        magma_int_t j, jj, jb, mj;
        for (j=0; j<n; j+=nb) {
            jj =  j/(nb*num_gpus);
            k  = (j/nb)%num_gpus;
            
            jb = min(nb, (n-j));
            mj = j+jb;
            
            magma_setdevice(k);
            magma_zsetmatrix_async( mj, jb,
                                     A(0, j),        lda,
                                    dA(k, 0, jj*nb), ldda,
                                    stream[k][0] );
        }
    }
    for( k=0; k<num_gpus; k++ ) {
        magma_setdevice(k);
        magma_queue_sync(stream[k][0]);
    }
    magma_setdevice(0);
    
    return MAGMA_SUCCESS;
}

extern "C" void
magma_zher2k_mgpu(int num_gpus, char uplo, char trans, int nb, int n, int k,
    magmaDoubleComplex alpha, magmaDoubleComplex **db, int lddb, int offset_b,
    double beta,           magmaDoubleComplex **dc, int lddc, int offset,
    int num_streams, magma_queue_t stream[][10])
{

#define dB(id, i, j)  (db[(id)]+(j)*lddb + (i)+offset_b)
#define dB1(id, i, j) (db[(id)]+(j)*lddb + (i)+offset_b)+k*lddb
#define dC(id, i, j)  (dc[(id)]+(j)*lddc + (i))

    char uplo_[2]  = {uplo, 0};
    int i, id, ib, ii, kk, n1, m1;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;

    /* diagonal update */
    for( i=0; i<n; i+=nb ) {
        id = ((i+offset)/nb)%num_gpus;
        kk = (i/(nb*num_gpus))%num_streams;
        magma_setdevice(id);
        magmablasSetKernelStream(stream[id][kk]);

        ib = min(nb, n-i);
        ii = nb*((i+offset)/(nb*num_gpus));

        /* zher2k on diagonal block */
        trace_gpu_start( id, kk, "syr2k", "syr2k" );
        magma_zher2k(uplo, trans, ib, k,
                     alpha, dB1(id, i,        0 ), lddb,
                            dB(id,  i,        0 ), lddb,
                     beta,  dC(id,  i+offset,   ii), lddc);
        trace_gpu_end( id, kk );
    }

    /* off-diagonal update */
    if( lapackf77_lsame( uplo_, "U" ) ) {
        for( i=nb; i<n; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%num_streams;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            
            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, i, ib, k,
                        alpha, dB1(id, 0, 0 ), lddb,
                               dB(id,  i, 0 ), lddb,
                        c_one, dC(id,  0, ii), lddc);
        }
    }
    else {
        for( i=0; i<n-nb; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%num_streams;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            
            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            n1 = n-i-ib;
            
            // zgemm on off-diagonal blocks
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                        alpha, dB1(id, i+ib,        0 ), lddb,
                               dB(id,  i,           0 ), lddb,
                        c_one, dC(id,  i+offset+ib, ii), lddc);
            trace_gpu_end( id, kk );
        }
    }

    if( lapackf77_lsame( uplo_, "U" ) ) {
        for( i=nb; i<n; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%num_streams;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            
            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, i, ib, k,
                        alpha, dB( id, 0, 0 ), lddb,
                               dB1(id, i, 0 ), lddb,
                        c_one, dC(id,  0, ii), lddc);
        }
    } else {
        for( i=0; i<n-nb; i+=nb ) {
            id = ((i+offset)/nb)%num_gpus;
            kk = (i/(nb*num_gpus))%num_streams;
            magma_setdevice(id);
            magmablasSetKernelStream(stream[id][kk]);
            
            ib = min(nb, n-i);
            ii = nb*((i+offset)/(nb*num_gpus));
            n1 = n-i-ib;
            
            /* zgemm on off-diagonal blocks */
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm(MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                        alpha, dB(id,  i+ib,        0 ), lddb,
                               dB1(id, i,           0 ), lddb,
                        c_one, dC(id,  i+offset+ib, ii), lddc);
            trace_gpu_end( id, kk );
        }
    }

    for( id=0; id<num_gpus; id++ ) {
        magma_setdevice(id);
        for( kk=0; kk<num_streams; kk++ ) magma_queue_sync(stream[id][kk]);
        magmablasSetKernelStream(NULL);
    }
}

#endif /* GPUSHMEM >= 200 */
