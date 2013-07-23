/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "../testing/flops.h"


#define A(i, j)  (a   +(j)*lda  + (i))
#define dA(d, i, j) (dwork[(d)]+(j)*lddla + (i))
#define dT(d, i, j) (dt[(d)]   +(j)*ldda  + (i))
#define dAup(d, i, j) (dwork[(d)]+(j)*NB + (i))
#define dTup(d, i, j) (dt[(d)]   +(j)*nb + (i))

extern "C" magma_int_t
magma_zpotrf_m(magma_int_t num_gpus0, char uplo, magma_int_t n,
               magmaDoubleComplex *a, magma_int_t lda, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZPOTRF_OOC computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine. The matrix A may not fit entirely in the GPU memory.

    The factorization has the form
       A = U**H * U,  if UPLO = 'U', or
       A = L  * L**H, if UPLO = 'L',
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    =========
    UPLO    (input) CHARACTER*1
            = 'U':  Upper triangle of A is stored;
            = 'L':  Lower triangle of A is stored.

    N       (input) INTEGER
            The order of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the symmetric matrix A.  If UPLO = 'U', the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = 'L', the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.

            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U or A = L * L**H.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    =====================================================================    */


    /* Local variables */
    double                 d_one     =  1.0;
    double                 d_neg_one = -1.0;
    magmaDoubleComplex        c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex        c_neg_one = MAGMA_Z_NEG_ONE;
    char                   uplo_[2]  = {uplo, 0};
    int                    upper     = lapackf77_lsame(uplo_, "U");

    magmaDoubleComplex *dwork[MagmaMaxGPUs], *dt[MagmaMaxGPUs], *work;
    magma_int_t     ldda, lddla, ldwrk, nb, iinfo, n_local[MagmaMaxGPUs], J2, d, num_gpus;
    magma_int_t     j, jj, jb, jb1, jb2, jb3, J, JB, NB, MB, h;
    magma_queue_t    stream[MagmaMaxGPUs][3];
    magma_event_t     event[MagmaMaxGPUs][5];
//#define ROW_MAJOR_PROFILE
#ifdef ROW_MAJOR_PROFILE
    magma_timestr_t start, end, start0, end0;
    double chol_time = 1.0;
#endif
    *info = 0;
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    nb = magma_get_dpotrf_nb(n);
    if( num_gpus0 > n/nb ) {
        num_gpus = n/nb;
        if( n%nb != 0 ) num_gpus ++;
    } else {
        num_gpus = num_gpus0;
    }
    ldda  = ((n+31)/32)*32;
    lddla = ((nb*(1+n/(nb*num_gpus))+31)/32)*32;

    /* figure out NB */
    size_t freeMem, totalMem;
    cudaMemGetInfo( &freeMem, &totalMem );
    freeMem /= sizeof(magmaDoubleComplex);
    
    MB = n;  /* number of rows in the big panel    */
    NB = (magma_int_t)(num_gpus*((0.8*freeMem-2*nb*ldda)/lddla)); /* number of columns in the big panel */
    //NB = min(5*nb,n);

    if( NB >= n ) {
        #ifdef CHECK_ZPOTRF_OOC
        printf( "      * still fit in GPU memory.\n" );
        #endif
        NB = n;
    } else {
        #ifdef CHECK_ZPOTRF_OOC
        printf( "      * don't fit in GPU memory.\n" );
        #endif
        NB = (NB/nb) * nb;   /* making sure it's devisable by nb   */
    }
    #ifdef CHECK_ZPOTRF_OOC
    if( NB != n ) printf( "      * running in out-core mode (n=%d, NB=%d, nb=%d).\n",n,NB,nb );
    else          printf( "      * running in in-core mode  (n=%d, NB=%d, nb=%d).\n",n,NB,nb );
    fflush(stdout);
    #endif
    for (d=0; d<num_gpus; d++ ) {
        magma_setdevice(d);
        if (MAGMA_SUCCESS != magma_zmalloc( &dt[d], NB*lddla + max(2,num_gpus)*nb*ldda )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        dwork[d] = &dt[d][max(2,num_gpus)*nb*ldda];
        
        for( j=0; j<3; j++ )
            magma_queue_create( &stream[d][j] );
        for( j=0; j<5; j++ )
            magma_event_create( &event[d][j]  );
    }
#ifdef ROW_MAJOR_PROFILE
    start0 = get_current_time();
#endif
    magma_setdevice(0);
    ldwrk = n;
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &work, ldwrk*nb )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    if (nb <= 1 || nb >= n) {
        lapackf77_zpotrf(uplo_, &n, a, &lda, info);
    } else {

    /* Use hybrid blocked code. */
    if (upper) {
        /* =========================================================== *
         * Compute the Cholesky factorization A = U'*U.                *
         * big panel is divided by block-row and distributed in block  *
         * column cyclic format                                        */
        
        /* for each big-panel */
        for( J=0; J<n; J+=NB ) {
            JB = min(NB,n-J);
            jb = min(JB,nb);
            if( num_gpus0 > (n-J)/nb ) {
                num_gpus = (n-J)/nb;
                if( (n-J)%nb != 0 ) num_gpus ++;
            } else {
                num_gpus = num_gpus0;
            }
            h  = (JB+nb-1)/nb;
            
            /* load the new big-panel by block-rows */
            magma_zhtodpo( num_gpus, &uplo, JB, n, J, J, nb, a, lda, dwork, NB, stream, &iinfo);
            
#ifdef ROW_MAJOR_PROFILE
            start = get_current_time();
#endif      
            /* update with the previous big-panels */
            for( j=0; j<J; j+=nb ) {
                /* upload the diagonal of the block column (broadcast to all GPUs) */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_zsetmatrix_async( nb, JB,
                                            A(j, J),       lda,
                                            dTup(d, 0, J), nb,
                                            stream[d][0] );
                    n_local[d] = 0;
                }
                
                /* distribute off-diagonal blocks to GPUs */
                for( jj=J+JB; jj<n; jj+=nb ) {
                    d  = ((jj-J)/nb)%num_gpus;
                    magma_setdevice(d);
                    
                    jb2 = min(nb, n-jj);
                    magma_zsetmatrix_async( nb, jb2,
                                            A(j, jj),                    lda,
                                            dTup(d, 0, J+JB+n_local[d]), nb,
                                            stream[d][0] );
                    n_local[d] += jb2;
                }
                
                /* wait for the communication */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( stream[d][0] );
                }
                
                /* update the current big-panel using the previous block-row */
                jb3 = nb; //min(nb,J-j); // number of columns in this previous block-column (nb)
                for( jj=0; jj<JB; jj+=nb ) { /* diagonal */
                    d  = (jj/nb)%num_gpus;
                    J2 = jj/(nb*num_gpus);
                    
                    magma_setdevice(d);
                    magmablasSetKernelStream(stream[d][J2%2]);
                    
                    J2 = nb*J2;
                    jb1 = min(JB,jj+nb); // first row in the next block-row
                    jb2 = min(nb,JB-jj); // number of rows in this current block-row
                    jb  = jj; //jb1-jb2;       // number of columns in the off-diagona blocks (jj)
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                 jb, jb2, nb,
                                 c_neg_one, dTup(d, 0, J   ),  nb,
                                            dTup(d, 0, J+jb),  nb,
                                 c_one,     dAup(d, 0, J2), NB);
                    
                    magma_zherk(MagmaUpper, MagmaConjTrans, jb2, jb3,
                                d_neg_one, dTup(d, 0,  J+jb),  nb,
                                d_one,     dAup(d, jb, J2), NB);
                }
                
                if( n > J+JB ) { /* off-diagonal */
                    for( d=0; d<num_gpus; d++ ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][2]);
                        
                        /* local number of columns in the big panel */
                        n_local[d] = (((n-J)/nb)/num_gpus)*nb;
                        if (d < ((n-J)/nb)%num_gpus)
                            n_local[d] += nb;
                        else if (d == ((n-J)/nb)%num_gpus)
                            n_local[d] += (n-J)%nb;
                        
                        /* local number of columns in diagonal */
                        n_local[d] -= ((JB/nb)/num_gpus)*nb;
                        if (d < (JB/nb)%num_gpus)
                            n_local[d] -= nb;
                        
                        J2 = nb*(JB/(nb*num_gpus));
                        if( d < (JB/nb)%num_gpus ) J2+=nb;
                        
                        magma_zgemm( MagmaConjTrans, MagmaNoTrans,
                                     JB, n_local[d], nb,
                                     c_neg_one, dTup(d, 0, J   ),  nb,
                                                dTup(d, 0, J+JB),  nb,
                                     c_one,     dAup(d, 0, J2), NB);
                    }
                }
                
                /* wait for the previous updates */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( stream[d][0] );
                    magma_queue_sync( stream[d][1] );
                    magma_queue_sync( stream[d][2] );
                    magmablasSetKernelStream(NULL);
                }
                magma_setdevice(0);
            } /* end of updates with previous rows */
            
            /* factor the big panel */
            // using two streams
            //magma_zpotrf2_mgpu(num_gpus, uplo, JB, n-J, J, J, nb,
            //                   dwork, NB, dt, ldda, a, lda, h, stream, event, &iinfo);
            // using three streams
            magma_zpotrf3_mgpu(num_gpus, uplo, JB, n-J, J, J, nb,
                               dwork, NB, dt, ldda, a, lda, h, stream, event, &iinfo);
            if( iinfo != 0 ) {
                *info = J+iinfo;
                break;
            }
#ifdef ROW_MAJOR_PROFILE
            end = get_current_time();
            chol_time += GetTimerValue(start, end);
#endif      
            
            /* upload the off-diagonal (and diagonal!!!) big panel */
            magma_zdtohpo(num_gpus, &uplo, JB, n, J, J, nb, NB, a, lda, dwork, NB, stream, &iinfo);
        }
    } else {
        /* ========================================================= *
         * Compute the Cholesky factorization A = L*L'.              */
        
        /* for each big-panel */
        for( J=0; J<n; J+=NB ) {
            JB = min(NB,n-J);
            if( num_gpus0 > (n-J)/nb ) {
                num_gpus = (n-J)/nb;
                if( (n-J)%nb != 0 ) num_gpus ++;
            } else {
                num_gpus = num_gpus0;
            }
            h  = (JB+nb-1)/nb;
            
            /* load the new big-panel by block-columns */
            magma_zhtodpo( num_gpus, &uplo, n, JB, J, J, nb, a, lda, dwork, lddla, stream, &iinfo);
            
            /* update with the previous big-panels */
#ifdef ROW_MAJOR_PROFILE
            start = get_current_time();
#endif      
            for( j=0; j<J; j+=nb ) {
                /* upload the diagonal of big panel */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_zsetmatrix_async( JB, nb,
                                            A(J, j),     lda,
                                            dT(d, J, 0), ldda,
                                            stream[d][0] );
                    n_local[d] = 0;
                }
                
                /* upload off-diagonals */
                for( jj=J+JB; jj<n; jj+=nb ) {
                    d  = ((jj-J)/nb)%num_gpus;
                    magma_setdevice(d);
                    
                    jb2 = min(nb, n-jj);
                    magma_zsetmatrix_async( jb2, nb,
                                            A(jj, j),                  lda,
                                            dT(d, J+JB+n_local[d], 0), ldda,
                                            stream[d][0] );
                    n_local[d] += jb2;
                }
                
                /* wait for the communication */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( stream[d][0] );
                }
                
                /* update the current big-panel using the previous block-row */
                jb3 = nb; //min(nb,J-j);
                for( jj=0; jj<JB; jj+=nb ) { /* diagonal */
                    d  = (jj/nb)%num_gpus;
                    J2 = jj/(nb*num_gpus);
                    
                    magma_setdevice(d);
                    magmablasSetKernelStream(stream[d][J2%2]);
                    
                    J2 = nb*J2;
                    jb1 = min(JB,jj+nb);
                    jb2 = min(nb,JB-jj);
                    jb  = jj; //jb1-jb2;
                    magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                 jb2, jb, nb,
                                 c_neg_one, dT(d, J+jb, 0), ldda,
                                            dT(d, J,    0), ldda,
                                 c_one,     dA(d, J2,   0), lddla);
                    
                    magma_zherk(MagmaLower, MagmaNoTrans, jb2, jb3,
                                d_neg_one, dT(d, J+jb, 0),  ldda,
                                d_one,     dA(d, J2,  jb ), lddla);
                }
                
                if( n > J+JB ) { /* off-diagonal */
                    for( d=0; d<num_gpus; d++ ) {
                        magma_setdevice(d);
                        magmablasSetKernelStream(stream[d][2]);
                        
                        /* local number of columns in the big panel */
                        n_local[d] = (((n-J)/nb)/num_gpus)*nb;
                        if (d < ((n-J)/nb)%num_gpus)
                            n_local[d] += nb;
                        else if (d == ((n-J)/nb)%num_gpus)
                            n_local[d] += (n-J)%nb;
                        
                        /* local number of columns in diagonal */
                        n_local[d] -= ((JB/nb)/num_gpus)*nb;
                        if (d < (JB/nb)%num_gpus)
                            n_local[d] -= nb;
                        
                        J2 = nb*(JB/(nb*num_gpus));
                        if( d < (JB/nb)%num_gpus ) J2+=nb;
                        
                        magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                     n_local[d], JB, nb,
                                     c_neg_one, dT(d, J+JB, 0), ldda,
                                                dT(d, J,    0), ldda,
                                     c_one,     dA(d, J2,   0), lddla);
                    }
                }
                /* wait for the previous updates */
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    magma_queue_sync( stream[d][0] );
                    magma_queue_sync( stream[d][1] );
                    magma_queue_sync( stream[d][2] );
                    magmablasSetKernelStream(NULL);
                }
                magma_setdevice(0);
            }
            
            /* factor the big panel */
            // using two streams
            //magma_zpotrf2_mgpu(num_gpus, uplo, n-J, JB, J, J, nb,
            //                   dwork, lddla, dt, ldda, a, lda, h, stream, event, &iinfo);
            // using three streams
            magma_zpotrf3_mgpu(num_gpus, uplo, n-J, JB, J, J, nb,
                               dwork, lddla, dt, ldda, a, lda, h, stream, event, &iinfo);
            if( iinfo != 0 ) {
                *info = J+iinfo;
                break;
            }
#ifdef ROW_MAJOR_PROFILE
            end = get_current_time();
            chol_time += GetTimerValue(start, end);
#endif      
            /* upload the off-diagonal big panel */
            magma_zdtohpo( num_gpus, &uplo, n, JB, J, J, nb, JB, a, lda, dwork, lddla, stream, &iinfo);
        
        } /* end of for J */
    } /* if upper */
    } /* if nb */
#ifdef ROW_MAJOR_PROFILE
    end0 = get_current_time();
#endif
    if( num_gpus0 > n/nb ) {
        num_gpus = n/nb;
        if( n%nb != 0 ) num_gpus ++;
    } else {
        num_gpus = num_gpus0;
    }
    for (d=0; d<num_gpus; d++ ) {
        magma_setdevice(d);

        for( j=0; j<3; j++ ) {
            magma_queue_destroy( stream[d][j] );
        }
        magma_free( dt[d] );

        for( j=0; j<5; j++ ) {
            magma_event_destroy( event[d][j] );
        }
    }
    magma_setdevice(0);
    magma_free_pinned( work );

#ifdef ROW_MAJOR_PROFILE
    printf("\n n=%d NB=%d nb=%d\n",n,NB,nb);
    printf(" Without memory allocation: %f / %f = %f GFlop/s\n",
           FLOPS_ZPOTRF(n)/1000000, GetTimerValue(start0, end0),
           FLOPS_ZPOTRF(n)/(1000000*GetTimerValue(start0, end0)));
    printf(" Performance %f / %f = %f GFlop/s\n",
           FLOPS_ZPOTRF(n)/1000000, chol_time,
           FLOPS_ZPOTRF(n)/(1000000*chol_time));
#endif
    return *info;
} /* magma_zpotrf_ooc */

#undef A
#undef dA
#undef dT
#undef dAup
#undef dTup
