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


extern "C" magma_int_t
magma_zgetrf2_mgpu(magma_int_t num_gpus,
                   magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
                   magmaDoubleComplex **d_lAT, magma_int_t lddat, magma_int_t *ipiv,
                   magmaDoubleComplex **d_lAP, magmaDoubleComplex *a, magma_int_t lda,
                   magma_queue_t streaml[][2], magma_int_t *info);

extern "C" magma_int_t
magma_zgetrf_m(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
               magma_int_t *ipiv, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    ZGETRF_m computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.  This version does not
    require work space on the GPU passed as input. GPU memory is allocated
    in the routine. The matrix may not fit entirely in the GPU memory.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Note: The factorization of big panel is done calling multiple-gpu-interface.
    Pivots are applied on GPU within the big panel.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    =====================================================================    */

#define    A(i,j) (a   + (j)*lda + (i))
#define inAT(d,i,j) (dAT[d] + (i)*nb*ldn_local + (j)*nb)
#define inPT(d,i,j) (dPT[d] + (i)*nb*nb + (j)*nb*maxm)

//#define PROFILE
#ifdef PROFILE
    double flops, time_rmajor = 0, time_rmajor2 = 0, time_rmajor3 = 0, time_mem = 0;
    magma_timestr_t start, start1, start2, end1, end, start0 = get_current_time();
#endif
    magmaDoubleComplex    *dAT[MagmaMaxGPUs], *dA[MagmaMaxGPUs], *dPT[MagmaMaxGPUs];
    magmaDoubleComplex    c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex    c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t        iinfo = 0, nb, nbi, maxm, n_local[MagmaMaxGPUs], ldn_local;
    magma_int_t        N, M, NB, NBk, I, d, num_gpus;
    magma_int_t        i, ii, jj, h, offset, ib, rows, s;
    
    magma_queue_t stream[MagmaMaxGPUs][2];
    magma_event_t  event[MagmaMaxGPUs][2];

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* initialize nb */
    nb = magma_get_zgetrf_nb(m);
    maxm = ((m  + 31)/32)*32;

    /* figure out NB */
    size_t freeMem, totalMem;
    cudaMemGetInfo( &freeMem, &totalMem );
    freeMem /= sizeof(magmaDoubleComplex);
    
    /* number of columns in the big panel */
    NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
    char * ngr_nb_char = getenv("MAGMA_NGR_NB");
    if( ngr_nb_char != NULL ) NB = max( nb, min( NB, atoi(ngr_nb_char) ) );
    //NB = 5*max(nb,32);

    if( num_gpus0 > ceil((double)NB/nb) ) {
        num_gpus = (int)ceil((double)NB/nb);
    } else {
        num_gpus = num_gpus0;
    }
    if( num_gpus*NB >= n ) {
        #ifdef CHECK_ZGETRF_OOC
        printf( "      * still fit in GPU memory.\n" );
        #endif
        NB = n;
    } else {
        #ifdef CHECK_ZGETRF_OOC
        printf( "      * don't fit in GPU memory.\n" );
        #endif
        NB = num_gpus*NB;
        NB = max(nb,(NB / nb) * nb); /* making sure it's devisable by nb (x64) */
    }
    h = 1+(2+num_gpus);

    #ifdef CHECK_ZGETRF_OOC
    if( NB != n ) printf( "      * running in out-core mode (n=%d, NB=%d, nb=%d).\n",n,NB,nb );
    else          printf( "      * running in in-core mode  (n=%d, NB=%d, nb=%d).\n",n,NB,nb );
    fflush(stdout);
    #endif

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code for scalar of one tile. */
        lapackf77_zgetrf(&m, &n, a, &lda, ipiv, info);
    } else {
        /* Use hybrid blocked code. */

    /* allocate memory on GPU to store the big panel */
#ifdef PROFILE
    start = get_current_time();
#endif
    n_local[0] = (NB/nb)/num_gpus;
    if( NB%(nb*num_gpus) != 0 ) n_local[0] ++;
    n_local[0] *= nb;
    ldn_local = ((n_local[0]+31)/32)*32;

    for( d=0; d<num_gpus; d++ ) {
        magma_setdevice(d);
        if (MAGMA_SUCCESS != magma_zmalloc( &dA[d], (h*nb + ldn_local)*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        dPT[d] = dA[d] + nb*maxm;      /* for storing the previous panel from CPU */
        dAT[d] = dA[d] + h*nb*maxm;    /* for storing the big panel               */
        magma_queue_create( &stream[d][0] );
        magma_queue_create( &stream[d][1] );
        magma_event_create( &event[d][0] );
        magma_event_create( &event[d][1] );
    }
    //magma_setdevice(0);

#ifdef PROFILE
    end = get_current_time();
    printf( " memory-allocation time: %e\n",GetTimerValue(start, end)/1000.0 );
    start = get_current_time();
#endif
    for( I=0; I<n; I+=NB ) {
        M = m;
        N = min( NB, n-I );       /* number of columns in this big panel             */
        s = min(max(m-I,0),N)/nb; /* number of small block-columns in this big panel */

        maxm = ((M + 31)/32)*32;
        if( num_gpus0 > ceil((double)N/nb) ) {
            num_gpus = (int)ceil((double)N/nb);
        } else {
            num_gpus = num_gpus0;
        }

        for( d=0; d<num_gpus; d++ ) {
            n_local[d] = ((N/nb)/num_gpus)*nb;
            if (d < (N/nb)%num_gpus)
                n_local[d] += nb;
            else if (d == (N/nb)%num_gpus)
                n_local[d] += N%nb;
        }
        ldn_local = ((n_local[0]+31)/32)*32;
        
#ifdef PROFILE
        start2 = get_current_time();
#endif
        /* upload the next big panel into GPU, transpose (A->A'), and pivot it */
        magmablas_zsetmatrix_transpose_mgpu(num_gpus, stream, A(0,I), lda,
                                            dAT, ldn_local, dA, maxm, M, N, nb);
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            magma_queue_sync( stream[d][0] );
            magma_queue_sync( stream[d][1] );
            magmablasSetKernelStream(NULL);
        }

#ifdef PROFILE
        start1 = get_current_time();
#endif
        /* == --------------------------------------------------------------- == */
        /* == loop around the previous big-panels to update the new big-panel == */
        for( offset = 0; offset<min(m,I); offset+=NB )
        {
            NBk = min( m-offset, NB );
            /* start sending the first tile from the previous big-panels to gpus */
            for( d=0; d<num_gpus; d++ ) {
                magma_setdevice(d);
                nbi  = min( nb, NBk );
                magma_zsetmatrix_async( (M-offset), nbi,
                                        A(offset,offset), lda,
                                        dA[d],            (maxm-offset), stream[d][0] );
                
                /* make sure the previous update finished */
                magmablasSetKernelStream(stream[d][0]);
                //magma_queue_sync( stream[d][1] );
                magma_queue_wait_event( stream[d][0], event[d][0] );
                
                /* transpose */
                magmablas_ztranspose2( inPT(d,0,0), nb, dA[d], maxm-offset, M-offset, nbi);
            }
            
            /* applying the pivot from the previous big-panel */
            for( d=0; d<num_gpus; d++ ) {
                magma_setdevice(d);
                magmablasSetKernelStream(stream[d][1]);
                magmablas_zpermute_long3( inAT(d,0,0), ldn_local, ipiv, NBk, offset );
            }
            
            /* == going through each block-column of previous big-panels == */
            for( jj=0, ib=offset/nb; jj<NBk; jj+=nb, ib++ )
            {
                ii   = offset+jj;
                rows = maxm - ii;
                nbi  = min( nb, NBk-jj );
                for( d=0; d<num_gpus; d++ ) {
                    magma_setdevice(d);
                    
                    /* wait for a block-column on GPU */
                    magma_queue_sync( stream[d][0] );
                    
                    /* start sending next column */
                    if( jj+nb < NBk ) {
                        magma_zsetmatrix_async( (M-ii-nb), min(nb,NBk-jj-nb),
                                                A(ii+nb,ii+nb), lda,
                                                dA[d],          (rows-nb), stream[d][0] );
                        
                        /* make sure the previous update finished */
                        magmablasSetKernelStream(stream[d][0]);
                        //magma_queue_sync( stream[d][1] );
                        magma_queue_wait_event( stream[d][0], event[d][(1+jj/nb)%2] );
                        
                        /* transpose next column */
                        magmablas_ztranspose2( inPT(d,0,(1+jj/nb)%2), nb, dA[d], rows-nb, M-ii-nb, nb);
                    }
                    
                    /* update with the block column */
                    magmablasSetKernelStream(stream[d][1]);
                    magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                                 n_local[d], nbi, c_one, inPT(d,0,(jj/nb)%2), nb, inAT(d,ib,0), ldn_local );
                    if( M > ii+nb ) {
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                            n_local[d], M-(ii+nb), nbi, c_neg_one, inAT(d,ib,0), ldn_local,
                            inPT(d,1,(jj/nb)%2), nb, c_one, inAT(d,ib+1,0), ldn_local );
                    }
                    magma_event_record( event[d][(jj/nb)%2], stream[d][1] );
                
                } /* end of for each block-columns in a big-panel */
            }
        } /* end of for each previous big-panels */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            magma_queue_sync( stream[d][0] );
            magma_queue_sync( stream[d][1] );
            magmablasSetKernelStream(NULL);
        }

        /* calling magma-gpu interface to panel-factorize the big panel */
        if( M > I ) {
            //magma_zgetrf1_mgpu(num_gpus, M-I, N, nb, I, dAT, ldn_local, ipiv+I, dA, &a[I*lda], lda,
            //                   (magma_queue_t **)stream, &iinfo);
            magma_zgetrf2_mgpu(num_gpus, M-I, N, nb, I, dAT, ldn_local, ipiv+I, dA, &a[I*lda], lda,
                               stream, &iinfo);
            if( iinfo < 0 ) {
                *info = iinfo;
                break;
            } else if( iinfo != 0 ) {
                *info = iinfo + I * NB;
                //break;
            }
            /* adjust pivots */
            for( ii=I; ii<min(I+N,m); ii++ )
                ipiv[ii] += I;
        }
#ifdef PROFILE
        end1 = get_current_time();
        time_rmajor  += GetTimerValue(start1, end1);
        time_rmajor3 += GetTimerValue(start2, end1);
        time_mem += (GetTimerValue(start2, end1)-GetTimerValue(start1, end1))/1000.0;
#endif
        /* download the current big panel to CPU */
        magmablas_zgetmatrix_transpose_mgpu(num_gpus, stream, dAT, ldn_local, A(0,I), lda, dA, maxm, M, N, nb);
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            magma_queue_sync( stream[d][0] );
            magma_queue_sync( stream[d][1] );
            magmablasSetKernelStream(NULL);
        }
#ifdef PROFILE
        end1 = get_current_time();
        time_rmajor2 += GetTimerValue(start1, end1);
#endif

    } /* end of for */

#ifdef PROFILE
    end = get_current_time();
    flops = FLOPS_ZGETRF( m, n ) / 1000000;
    printf(" NB=%d nb=%d\n",NB,nb);
    printf(" memcopy and transpose %e seconds\n",time_mem );
    printf(" total time %e seconds\n",GetTimerValue(start0,end)/1000.0);
    printf(" Performance %f GFlop/s, %f seconds without htod and dtoh\n",     flops / time_rmajor,  time_rmajor /1000.0);
    printf(" Performance %f GFlop/s, %f seconds with    htod\n",              flops / time_rmajor3, time_rmajor3/1000.0);
    printf(" Performance %f GFlop/s, %f seconds with    dtoh\n",              flops / time_rmajor2, time_rmajor2/1000.0);
    printf(" Performance %f GFlop/s, %f seconds without memory-allocation\n", flops / GetTimerValue(start, end), GetTimerValue(start,end)/1000.0);
#endif

    for( d=0; d<num_gpus0; d++ ) {
        magma_setdevice(d);
        magma_free( dA[d] );
        magma_event_destroy( event[d][0] );
        magma_event_destroy( event[d][1] );
        magma_queue_destroy( stream[d][0] );
        magma_queue_destroy( stream[d][1] );
        magmablasSetKernelStream(NULL);
    }
    magma_setdevice(0);
    
    }
    
    return *info;
} /* magma_zgetrf_m */


extern "C" magma_int_t
magma_zgetrf_piv(magma_int_t num_gpus0, magma_int_t m, magma_int_t n, magmaDoubleComplex *a, magma_int_t lda,
                 magma_int_t *ipiv, magma_int_t *info)
{
    magma_int_t nb, h = 2, num_gpus;
    magma_int_t NB, I, k1, k2, incx, minmn, maxm;

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0)
        return *info;

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* initialize nb */
    nb = magma_get_zgetrf_nb(m);
    maxm = ((m  + 31)/32)*32;

    /* figure out NB */
    size_t freeMem, totalMem;
    cudaMemGetInfo( &freeMem, &totalMem );
    freeMem /= sizeof(magmaDoubleComplex);

    /* number of columns in the big panel */
    NB = (magma_int_t)(0.8*freeMem/maxm-h*nb);
    char * ngr_nb_char = getenv("MAGMA_NGR_NB");
    if( ngr_nb_char != NULL ) NB = max( nb, min( NB, atoi(ngr_nb_char) ) );
    //NB = 5*max(nb,32);

    if( num_gpus0 > ceil((double)NB/nb) ) {
        num_gpus = (int)ceil((double)NB/nb);
    } else {
        num_gpus = num_gpus0;
    }
    if( num_gpus*NB >= n ) {
        #ifdef CHECK_ZGETRF_OOC
        printf( "      * still fit in GPU memory.\n" );
        #endif
        NB = n;
    } else {
        #ifdef CHECK_ZGETRF_OOC
        printf( "      * don't fit in GPU memory.\n" );
        #endif
        NB = num_gpus*NB;
        NB = max(nb,(NB / nb) * nb); /* making sure it's devisable by nb (x64) */
    }
    minmn = min(m,n);

    for( I=0; I<minmn-NB; I+=NB ) {
        k1 = 1+I+NB;
        k2 = minmn;
        incx = 1;
        lapackf77_zlaswp(&NB, &a[I*lda], &lda, &k1, &k2, ipiv, &incx);
    }

    return *info;
} /* magma_zgetrf_piv */


extern "C" magma_int_t
magma_zgetrf2_piv(magma_int_t m, magma_int_t n, magma_int_t start, magma_int_t end,
                  magmaDoubleComplex *a, magma_int_t lda, magma_int_t *ipiv, magma_int_t *info)
{
    magma_int_t I, k1, k2, nb, incx, minmn;

    *info = 0;

    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0)
        return MAGMA_ERR_ILLEGAL_VALUE;

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return MAGMA_SUCCESS;

    /* initialize nb */
    nb = magma_get_zgetrf_nb(m);
    minmn = min(end,min(m,n));

    for( I=start; I<end-nb; I+=nb ) {
        incx = 1;
        k1 = 1+I+nb;
        k2 = minmn;
        lapackf77_zlaswp(&nb, &a[I*lda], &lda, &k1, &k2, ipiv, &incx);
    }

    return MAGMA_SUCCESS;
} /* magma_zgetrf_piv */


#undef inAT
#undef inPT
#undef A

