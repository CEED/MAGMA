/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "trace.h"

/* === Define what BLAS to use ============================================ */
#define PRECISION_z

#if defined(PRECISION_s) || defined(PRECISION_d)
#define ZTRSM_WORK
//#define magma_ztrsm magmablas_ztrsm
#endif
/* === End defining what BLAS to use ======================================= */


#define Alo(i, j)  (a   +            ((j)+off_j)*lda  + (nb*(((i)/nb)%h)+off_i))
#define Aup(i, j)  (a   +(nb*(((j)/nb)%h)+off_j)*lda  +               (i+off_i))

#define dlA(id, i, j)     (d_lA[(id)] + (j)*ldda + (i))
#define dlP(id, i, j, k)  (d_lP[(id)] + (k)*nb*lddp + (j)*lddp + (i))
#define dlPT(id, i, j, k) (d_lP[(id)] + (k)*nb*lddp + (j)*nb   + (i))


extern "C" magma_int_t
magma_zpotrf3_mgpu(magma_int_t num_gpus, char uplo, magma_int_t m, magma_int_t n,
                   magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
                   magmaDoubleComplex **d_lA,  magma_int_t ldda,
                   magmaDoubleComplex **d_lP,  magma_int_t lddp,
                   magmaDoubleComplex *a,      magma_int_t lda,   magma_int_t h,
                   magma_queue_t stream[][3], magma_event_t event[][5],
                   magma_int_t *info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.
    Auxiliary subroutine for zpotrf2_ooc. It is multiple gpu interface to compute
    Cholesky of a "rectangular" matrix.

    The factorization has the form
       dA = U**H * U,  if UPLO = 'U', or
       dA = L  * L**H,  if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of dA is stored;
            = 'L':  Lower triangle of dA is stored.

    N       (input) INTEGER
            The order of the matrix dA.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the Hermitian matrix dA.  If UPLO = 'U', the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    LDDA     (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            dividable by 16.

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.
    =====================================================================   */


    magma_int_t     j, jb, nb0, nb2, d, dd, id, j_local, j_local2, buf;
    char            uplo_[2] = {uplo, 0};
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double          d_one     =  1.0;
    double          d_neg_one = -1.0;
    int upper = lapackf77_lsame(uplo_, "U");
    magmaDoubleComplex *dlpanel;
    magma_int_t n_local[MagmaMaxGPUs], ldpanel;
    //magma_event_t event0[MagmaMaxGPUs],  /* send row to CPU    */
    //            event1[MagmaMaxGPUs],  /* send diag to GPU   */
    //            event2[MagmaMaxGPUs],  /* offdiagonal update */
    //            event3[MagmaMaxGPUs],  /* send row to GPU    */
    //            event4[MagmaMaxGPUs];  /* lookahead          */
    const magma_int_t stream1 = 0, stream2 = 1, stream3 = 2;
    #ifdef ZTRSM_WORK
    magmaDoubleComplex *d_dinvA[MagmaMaxGPUs][2], *d_x[MagmaMaxGPUs][2]; /* used by ztrsm_work */
    #endif
    
    *info = 0;
    if ( (! upper) && (! lapackf77_lsame(uplo_, "L")) ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (!upper && num_gpus*ldda < max(1,n)) {
        *info = -4;
    } else if (upper && ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* initialization */
    for( d=0; d<num_gpus; d++ ) {
        /* local-n and local-ld */
        if (upper) {
            n_local[d] = ((n/nb)/num_gpus)*nb;
            if (d < (n/nb)%num_gpus)
                n_local[d] += nb;
            else if (d == (n/nb)%num_gpus)
                n_local[d] += n%nb;
        } else {
            n_local[d] = ((m/nb)/num_gpus)*nb;
            if (d < (m/nb)%num_gpus)
                n_local[d] += nb;
            else if (d == (m/nb)%num_gpus)
                n_local[d] += m%nb;
        }
        //magma_setdevice(d);
        //magma_event_create( &event0[d] );
        //magma_event_create( &event1[d] );
        //magma_event_create( &event2[d] );
        //magma_event_create( &event3[d] );
        //magma_event_create( &event4[d] );
    }

    /* == initialize the trace */
    trace_init( 1, num_gpus, 3, (CUstream_st**)stream );

    if (upper)
    {
        /* ---------------------------------------------- */
        /* Upper-triangular case                          */
        /* > Compute the Cholesky factorization A = U'*U. */
        /* ---------------------------------------------- */
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
        /* invert the diagonals
         * Allocate device memory for the inversed diagonal blocks, size=m*NB
         */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            for( j=0; j<2; j++ ) {
                magma_zmalloc( &d_dinvA[d][j], nb*nb );
                magma_zmalloc( &d_x[d][j],      n*nb );
                cudaMemset(d_dinvA[d][j], 0, nb*nb*sizeof(magmaDoubleComplex));
                cudaMemset(d_x[d][j],     0,  n*nb*sizeof(magmaDoubleComplex));
            }
        }
        magma_setdevice(0);
#endif

        for (j=0; j<m; j+=nb) {

            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%num_gpus;
            buf = (j/nb)%num_gpus;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*num_gpus);
            jb = min(nb, (m-j));
            
            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if( j > 0 ) {
                magmablasSetKernelStream(stream[id][stream1]);
                trace_gpu_start( id, stream1, "syrk", "syrk" );
                magma_zherk(MagmaUpper, MagmaConjTrans, jb, j,
                            d_neg_one, dlA(id, 0, nb*j_local), ldda,
                            d_one,     dlA(id, j, nb*j_local), ldda);
                trace_gpu_end( id, stream1 );
            }
            
            /* send the diagonal to cpu on stream1 */
            trace_gpu_start( id, stream1, "comm", "D to CPU" );
            magma_zgetmatrix_async( jb, jb,
                                    dlA(id, j, nb*j_local), ldda,
                                    Aup(j,j),               lda,
                                    stream[id][stream1] );
            trace_gpu_end( id, stream1 );

            /* update off-diagonal blocks in the panel */
            if( j > 0 ) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    j_local2 = j_local+1;
                    if( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if( n_local[d] > nb0 ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream2]);
                        if( d == id ) {
                            dlpanel = dlA(d, 0, nb*j_local);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlP(d, jb, 0, buf);
                            ldpanel = lddp;
                            magma_queue_wait_event( stream[d][stream2], event[d][0] ); // rows arrived at gpu
                        }
                        trace_gpu_start( d, stream2, "gemm", "gemm" );
                        magma_zgemm(MagmaConjTrans, MagmaNoTrans,
                                    jb, n_local[d]-nb0, j,
                                    c_neg_one, dlpanel,        ldpanel,
                                               dlA(d, 0, nb0), ldda,
                                    c_one,     dlA(d, j, nb0), ldda);
                        trace_gpu_end( d, stream2 );
                        magma_event_record( event[d][2], stream[d][stream2] );
                    }
                    d = (d+1)%num_gpus;
                }
            }

            /* wait for panel and factorize it on cpu */
            magma_setdevice(id);
            magma_queue_sync( stream[id][stream1] );
            trace_cpu_start( 0, "getrf", "getrf" );
            lapackf77_zpotrf(MagmaUpperStr, &jb, Aup(j,j), &lda, info);
            trace_cpu_end( 0 );
            if (*info != 0) {
                *info = *info + j;
                break;
            }
            
            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < n) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    if( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d, 0, 0, buf);
                        ldpanel = lddp;
                    }
                    magma_setdevice(d);
                    trace_gpu_start( d, stream1, "comm", "comm" );
                    magma_zsetmatrix_async( jb, jb,
                                            Aup(j,j), lda,
                                            dlpanel,  ldpanel,
                                            stream[d][stream1] );
                    trace_gpu_end( d, stream1 );
                    magma_event_record( event[d][1], stream[d][stream1] );
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_setdevice(id);
                trace_gpu_start( id, stream1, "comm", "comm" );
                magma_zsetmatrix_async( jb, jb,
                                        Aup(j,j),               lda,
                                        dlA(id, j, nb*j_local), ldda,
                                        stream[id][stream1] );
                trace_gpu_end( id, stream1 );
            }
            
            /* panel-factorize the off-diagonal */
            if ( (j+jb) < n) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d,j,nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d, 0, 0, buf);
                        ldpanel = lddp;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2);
                    
                    magma_setdevice(d);
                    //magma_queue_sync( stream[d][stream1] );  // synch on chol for remaining update
                    //magma_queue_sync( stream[d][stream2] );
                    if( j+jb < m && d == (j/nb+1)%num_gpus ) {
                        /* owns the next column, look-ahead next block on stream1 */
                        magma_queue_wait_event( stream[d][stream1], event[d][2] ); // wait for gemm update
                        magmablasSetKernelStream(stream[d][stream1]);
                        trace_gpu_start( d, stream1, "trsm", "trsm" );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                              jb, nb0, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              1, d_dinvA[d][0], d_x[d][0] );
#else
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb0, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        magma_event_record( event[d][4], stream[d][stream1] );
                        trace_gpu_end( d, stream1 );
                    } else if( nb2 > 0 ) {
                        /* update all the blocks on stream2 */
                        magma_queue_wait_event( stream[d][stream2], event[d][1] ); // wait for cholesky factor
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
                        magmablasSetKernelStream(stream[d][stream2]);
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, j, nb*j_local2), ldda,
                                              1, d_dinvA[d][1], d_x[d][1] );
#else
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, j, nb*j_local2), ldda);
#endif
                        trace_gpu_end( d, stream2 );
                    }
                    d = (d+1)%num_gpus;
                } /* end of for */

                /* ========================================================== */
                d = (j/nb+1)%num_gpus;
                /* next column */
                j_local2 = j_local+1;
                if( d > id ) j_local2--;
                nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                /* even on 1 gpu, off-diagonals are copied to cpu (synchronize at the end).      *
                 * so we have the Cholesky factor, but only diagonal submatrix of the big panel, *
                 * on cpu at the end.                                                            */
                if( j+jb < m ) {
                    int d2, id2, j2, buf2;
                    magma_setdevice(d);
                    /* make sure all the previous sets are done */
                    if( h < num_gpus ) {
                        /* > offdiagonal */
                        for( d2=0; d2<num_gpus; d2++ ) {
                            j2 = j - (1+d2)*nb;
                            if( j2 < 0 ) break;
                            id2  = (j2/nb)%num_gpus;
                            magma_queue_wait_event( stream[d][stream3], event[id2][0] );
                        }
                
                        /* > diagonal */
                        for( d2=0; d2<num_gpus; d2++ ) {
                            j2 = j - d2*nb;
                            if( j2 < 0 ) break;
                            id2  = (j2/nb)%num_gpus;
                            magma_queue_wait_event( stream[d][stream3], event[id2][1] );
                        }
                    }
                    /* lookahead */
                    magma_queue_wait_event( stream[d][stream3], event[d][4] );
                
                    trace_gpu_start( d, stream3, "comm", "row to CPU" );
                    magma_zgetmatrix_async( (j+jb), nb0,
                                            dlA(d, 0, nb*j_local2), ldda,
                                            Aup(0,j+jb),            lda,
                                            stream[d][stream3] );
                    trace_gpu_end( d, stream3 );
                    magma_event_record( event[d][3], stream[d][stream3] );
                    /* needed on pluto */
                    //magma_queue_sync( stream[d][stream3] );
                
                    /* wait for the off-diagonal on cpu */
                    //magma_setdevice(id);
                    //magma_queue_sync( stream[id][stream3] );
                
                    /* broadcast rows to gpus on stream2 */
                    buf2 = ((j+jb)/nb)%num_gpus;
                    for( d2=0; d2<num_gpus; d2++ ) {
                        if( d2 != d )
                        {
                            magma_setdevice(d2);
                            trace_gpu_start( d2, stream3, "comm", "row to GPUs" );
                            magma_queue_wait_event( stream[d2][stream3], event[d][3] ); // rows arrived at cpu on stream3
                            magma_zsetmatrix_async( j+jb, nb0,
                                                    Aup(0,j+jb),        lda,
                                                    dlP(d2,nb0,0,buf2), lddp,
                                                    stream[d2][stream3] );
                            trace_gpu_end( d2, stream3 );
                            magma_event_record( event[d2][0], stream[d2][stream3] );
                        }
                    }
                }

                /* ========================================================== */
                /* gpu owning the next column                    */
                /* after look ahead, update the remaining blocks */
                if( j+jb < m ) /* no update on the last block column */
                {
                    d = (j/nb+1)%num_gpus;
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d, j, nb*j_local);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlP(d, 0, 0, buf);
                        ldpanel = lddp;
                    }
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                    nb2 =         n_local[d]-nb*j_local2 - nb0;
                
                    /* update the remaining blocks */
                    if( nb2 > 0 ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream2]);
                        //magma_queue_wait_event( stream[d][stream2], event[d][1] ); // wait for cholesky factor
                        magma_queue_wait_event( stream[d][stream2], event[d][4] ); // lookahead -> diagonal inversion
                        trace_gpu_start( d, stream2, "trsm", "trsm" );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                              jb, nb2, c_one,
                                              dlpanel,                    ldpanel,
                                              dlA(d, j, nb*j_local2+nb0), ldda,
                                              0, d_dinvA[d][0], d_x[d][1] );
#else
                        magma_ztrsm( MagmaLeft, MagmaUpper, MagmaConjTrans, MagmaNonUnit,
                                     jb, nb2, c_one,
                                     dlpanel,                    ldpanel,
                                     dlA(d, j, nb*j_local2+nb0), ldda);
#endif
                        trace_gpu_end( d, stream2 );
                    }
                }
            } /* end of ztrsm */
        } /* end of for j=1, .., n */
    } else {

        /* ---------------------------------------------- */
        /* Lower-triangular case                          */
        /* > Compute the Cholesky factorization A = L*L'. */
        /* ---------------------------------------------- */
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
        /*
         * Allocate device memory for the inversed diagonal blocks, size=N*BLOCK_SIZE
         */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            for( j=0; j<2; j++ ) {
                magma_zmalloc( &d_dinvA[d][j], nb*nb );
                magma_zmalloc( &d_x[d][j],     nb*m  );
                cudaMemset(d_dinvA[d][j], 0, nb*nb*sizeof(magmaDoubleComplex));
                cudaMemset(d_x[d][j],     0, nb* m*sizeof(magmaDoubleComplex));
            }
        }
        magma_setdevice(0);
#endif

        for (j=0; j<n; j+=nb) {
        
            /* Set the GPU number that holds the current panel */
            id  = (j/nb)%num_gpus;
            buf = (j/nb)%num_gpus;
            
            /* Set the local index where the current panel is */
            j_local = j/(nb*num_gpus);
            jb = min(nb, (n-j));
            
            /* Update the current diagonal block on stream1 */
            magma_setdevice(id);
            if( j > 0 ) {
                magmablasSetKernelStream(stream[id][stream1]);
                magma_zherk(MagmaLower, MagmaNoTrans, jb, j,
                            d_neg_one, dlA(id, nb*j_local, 0), ldda,
                            d_one,     dlA(id, nb*j_local, j), ldda);
            }
            
            /* send the diagonal to cpu on stream1 */
            magma_zgetmatrix_async( jb, jb,
                                    dlA(id, nb*j_local, j), ldda,
                                    Alo(j,j),               lda,
                                    stream[id][stream1] );

            /* update off-diagonal blocks of the panel */
            if( j > 0 ) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    j_local2 = j_local+1;
                    if( d > id ) j_local2 --;
                    nb0 = nb*j_local2;
            
                    if( nb0 < n_local[d] ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream2]);
                        if( d == id ) {
                            dlpanel = dlA(d, nb*j_local, 0);
                            ldpanel = ldda;
                        } else {
                            dlpanel = dlPT(d,0,jb,buf);
                            ldpanel = nb;
                            magma_queue_wait_event( stream[d][stream2], event[d][0] ); // rows arrived at gpu
                        }
                        magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d]-nb0, jb, j,
                                     c_neg_one, dlA(d, nb0, 0), ldda,
                                                dlpanel,        ldpanel,
                                     c_one,     dlA(d, nb0, j), ldda);
                        magma_event_record( event[d][2], stream[d][stream2] );
                    }
                    d = (d+1)%num_gpus;
                }
            }

            /* wait for the panel and factorized it on cpu */
            magma_setdevice(id);
            magma_queue_sync( stream[id][stream1] );
            lapackf77_zpotrf(MagmaLowerStr, &jb, Alo(j,j), &lda, info);
            if (*info != 0) {
                *info = *info + j;
                break;
            }

            /* send the diagonal to gpus on stream1 */
            if ( (j+jb) < m) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    if( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    magma_setdevice(d);
                    magma_zsetmatrix_async( jb, jb,
                                            Alo(j,j), lda,
                                            dlpanel,  ldpanel,
                                            stream[d][stream1] );
                    magma_event_record( event[d][1], stream[d][stream1] );
                    d = (d+1)%num_gpus;
                }
            } else {
                magma_setdevice(id);
                magma_zsetmatrix_async( jb, jb,
                                        Alo(j,j),               lda,
                                        dlA(id, nb*j_local, j), ldda,
                                        stream[id][stream1] );
            }

            /* panel factorize the off-diagonal */
            if ( (j+jb) < m) {
                d = (j/nb+1)%num_gpus;
                for( dd=0; dd<num_gpus; dd++ ) {
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d, 0, 0, buf);
                        ldpanel = nb;
                    }
                    nb2 = n_local[d] - j_local2*nb;
                    nb0 = min(nb, nb2 );
                    
                    magma_setdevice(d);
                    if( j+nb < n && d == (j/nb+1)%num_gpus ) { /* owns next column, look-ahead next block on stream1 */
                        magma_queue_wait_event( stream[d][stream1], event[d][2] ); // wait for gemm update
                        magmablasSetKernelStream(stream[d][stream1]);
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                              nb0, jb, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              1, d_dinvA[d][0], d_x[d][0] );
#else
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                     nb0, jb, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                        magma_event_record( event[d][4], stream[d][stream1] );
                    } else if( nb2 > 0 ) { /* other gpus updating all the blocks on stream2 */
                        /* update the entire column */
                        magma_queue_wait_event( stream[d][stream2], event[d][1] ); // wait for the cholesky factor
                        magmablasSetKernelStream(stream[d][stream2]);
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                ldpanel,
                                              dlA(d, nb*j_local2, j), ldda,
                                              1, d_dinvA[d][1], d_x[d][1] );
#else
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                ldpanel,
                                     dlA(d, nb*j_local2, j), ldda);
#endif
                    }
                    d = (d+1)%num_gpus;
                } /* end for d */

                /* ========================================================== */
                d = (j/nb+1)%num_gpus;
                /* next column */
                j_local2 = j_local+1;
                if( d > id ) j_local2--;
                nb0 = min(nb, n_local[d]-nb*j_local2 );
                
                /* even on 1 gpu, we copy off-diagonal to cpu (but don't synchronize).  */
                /* so we have the Cholesky factor on cpu at the end.                    */
                if( j+jb < n ) {
                    int d2, id2, j2, buf2;
                    magma_setdevice(d);
                    /* make sure all the previous sets are done */
                    if( h < num_gpus ) {
                        /* > offdiagonal */
                        for( d2=0; d2<num_gpus; d2++ ) {
                            j2 = j - (1+d2)*nb;
                            if( j2 < 0 ) break;
                            id2  = (j2/nb)%num_gpus;
                            magma_queue_wait_event( stream[d][stream3], event[id2][0] );
                        }
                
                        /* > diagonal */
                        for( d2=0; d2<num_gpus; d2++ ) {
                            j2 = j - d2*nb;
                            if( j2 < 0 ) break;
                            id2  = (j2/nb)%num_gpus;
                            magma_queue_wait_event( stream[d][stream3], event[id2][1] );
                        }
                    }
                    // lookahead done
                    magma_queue_wait_event( stream[d][stream3], event[d][4] );
                
                    magma_zgetmatrix_async( nb0, j+jb,
                                            dlA(d, nb*j_local2, 0), ldda,
                                            Alo(j+jb,0),            lda,
                                            stream[d][stream3] );
                    magma_event_record( event[d][3], stream[d][stream3] );
                    /* syn on rows on CPU, seem to be needed on Pluto */
                    //magma_queue_sync( stream[d][stream3] );
                
                    /* broadcast the rows to gpus */
                    buf2 = ((j+jb)/nb)%num_gpus;
                    for( d2=0; d2<num_gpus; d2++ ) {
                        if( d2 != d )
                        {
                            magma_setdevice(d2);
                            magma_queue_wait_event( stream[d2][stream3], event[d][3] ); // getmatrix done
                            magma_zsetmatrix_async( nb0, j+jb,
                                                    Alo(j+jb,0),         lda,
                                                    dlPT(d2,0,nb0,buf2), nb,
                                                    stream[d2][stream3] );
                            magma_event_record( event[d2][0], stream[d2][stream3] );
                        }
                    }
                }

                /* ========================================================== */
                /* gpu owning the next column updates remaining blocks on stream2 */
                if( j+nb < n ) { // no lookahead on the last block column
                    d = (j/nb+1)%num_gpus;
                    
                    /* next column */
                    j_local2 = j_local+1;
                    if( d > id ) j_local2--;
                    if( d == id ) {
                        dlpanel = dlA(d, nb*j_local, j);
                        ldpanel = ldda;
                    } else {
                        dlpanel = dlPT(d,0,0,buf);
                        ldpanel = nb;
                    }
                    nb0 = min(nb, n_local[d]-nb*j_local2 );
                    nb2 = n_local[d] - j_local2*nb - nb0;

                    if( nb2 > 0 ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][stream2]);
                        /* update the remaining blocks in the column */
                        //magma_queue_wait_event( stream[d][stream2], event[d][1] ); // panel received
                        magma_queue_wait_event( stream[d][stream2], event[d][4] ); // lookahead -> diagonal inversion
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
                        magmablas_ztrsm_work( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                              nb2, jb, c_one,
                                              dlpanel,                    ldpanel,
                                              dlA(d, nb*j_local2+nb0, j), ldda,
                                              0, d_dinvA[d][0], d_x[d][1] );
#else
                        magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                                     nb2, jb, c_one,
                                     dlpanel,                    ldpanel,
                                     dlA(d, nb*j_local2+nb0, j), ldda);
#endif
                    }
                }
            }
        }
    } /* end of else not upper */

    /* == finalize the trace == */
    trace_finalize( "zpotrf.svg","trace.css" );
    for( d=0; d<num_gpus; d++ ) {
        magma_setdevice(d);
        magma_queue_sync( stream[d][0] );
        magma_queue_sync( stream[d][1] );
        magma_queue_sync( stream[d][2] );
        magmablasSetKernelStream(NULL);
        
        //magma_event_destroy( event0[d] );
        //magma_event_destroy( event1[d] );
        //magma_event_destroy( event2[d] );
        //magma_event_destroy( event3[d] );
        //magma_event_destroy( event4[d] );
#if (defined(PRECISION_d) || defined(PRECISION_s)) && defined(ZTRSM_WORK)
        for( j=0; j<2; j++ ) {
            magma_free( d_dinvA[d][j] );
            magma_free( d_x[d][j] );
        }
#endif
    }
    magma_setdevice(0);

    return *info;
} /* magma_zpotrf_mgpu */

#undef Alo
#undef Aup
#define A(i, j)  (a +(j)*lda  + (i))
#define dA(d, i, j) (dwork[(d)]+(j)*ldda + (i))

extern "C" magma_int_t
magma_zhtodpo(magma_int_t num_gpus, char *uplo, magma_int_t m, magma_int_t n,
              magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
              magmaDoubleComplex *a,      magma_int_t lda,
              magmaDoubleComplex **dwork, magma_int_t ldda,
              magma_queue_t stream[][3], magma_int_t *info)
{
    magma_int_t k;

    if( lapackf77_lsame(uplo, "U") ) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j; j<n; j+=nb) {
            jj = (j-off_j)/(nb*num_gpus);
            k  = ((j-off_j)/nb)%num_gpus;
            magma_setdevice(k);
            
            jb = min(nb, (n-j));
            if(j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;
            magma_zsetmatrix_async( mj, jb,
                                    A(off_i, j),     lda,
                                    dA(k, 0, jj*nb), ldda,
                                    stream[k][0] );
        }
    }
    else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for(i=off_i; i<m; i+=nb){
            ii = (i-off_i)/(nb*num_gpus);
            k  = ((i-off_i)/nb)%num_gpus;
            magma_setdevice(k);
            
            ib = min(nb, (m-i));
            if(i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_zsetmatrix_async( ib, ni,
                                    A(i, off_j),     lda,
                                    dA(k, ii*nb, 0), ldda,
                                    stream[k][0] );
        }
    }
    for( k=0; k<num_gpus; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( stream[k][0] );
    }
    magma_setdevice(0);

    return *info;
}


extern "C" magma_int_t
magma_zdtohpo(magma_int_t num_gpus, char *uplo, magma_int_t m, magma_int_t n,
              magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
              magmaDoubleComplex *a,      magma_int_t lda,
              magmaDoubleComplex **dwork, magma_int_t ldda,
              magma_queue_t stream[][3], magma_int_t *info)
{
    magma_int_t k, stream_id = 1;
    if( lapackf77_lsame(uplo, "U") ) {
        magma_int_t j, jj, jb, mj;
        
        /* go through each column */
        for (j=off_j+NB; j<n; j+=nb) {
            jj = (j-off_j)/(nb*num_gpus);
            k  = ((j-off_j)/nb)%num_gpus;
            magma_setdevice(k);
            
            jb = min(nb, (n-j));
            if(j+jb < off_j+m)
                mj = (j-off_i)+jb;
            else
                mj = m;
            magma_zgetmatrix_async( mj, jb,
                                    dA(k, 0, jj*nb), ldda,
                                    A(off_i, j),     lda,
                                    stream[k][stream_id] );
        }
    } else {
        magma_int_t i, ii, ib, ni;
        
        /* go through each row */
        for(i=off_i+NB; i<m; i+=nb){
            ii = (i-off_i)/(nb*num_gpus);
            k  = ((i-off_i)/nb)%num_gpus;
            magma_setdevice(k);
            
            ib = min(nb, (m-i));
            if(i+ib < off_i+n)
                ni = (i-off_i)+ib;
            else
                ni = n;
            
            magma_zgetmatrix_async( ib, ni,
                                    dA(k, ii*nb, 0), ldda,
                                    A(i, off_j),     lda,
                                    stream[k][stream_id] );
        }
    }
    for( k=0; k<num_gpus; k++ ) {
        magma_setdevice(k);
        magma_queue_sync( stream[k][stream_id] );
    }
    magma_setdevice(0);

    return *info;
}

#undef A
#undef dA
#undef dlA
#undef dlP
#undef dlPT
