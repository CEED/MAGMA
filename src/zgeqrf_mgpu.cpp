/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "common_magma.h"

extern "C" magma_int_t
magma_zgeqrf2_mgpu( magma_int_t num_gpus, magma_int_t m, magma_int_t n,
                    magmaDoubleComplex **dlA, magma_int_t ldda,
                    magmaDoubleComplex *tau,
                    magma_int_t *info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======
    ZGEQRF2_MGPU computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R. This is a GPU interface of the routine.

    Arguments
    =========
    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDDA,N)
            On entry, the M-by-N matrix dA.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).

    LDDA    (input) INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    TAU     (output) COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ===============
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).
    =====================================================================    */

    #define dlA(dev, i, j)   (dlA[dev] + (i) + (j)*(ldda))
    #define hpanel(i)        (hpanel + (i))

    // set to NULL to make cleanup easy: free(NULL) does nothing.
    magmaDoubleComplex *dwork[MagmaMaxGPUs]={NULL}, *dpanel[MagmaMaxGPUs]={NULL};
    magmaDoubleComplex *hwork=NULL, *hpanel=NULL;
    magma_queue_t stream[MagmaMaxGPUs][2]={{NULL}};
    magma_event_t panel_event[MagmaMaxGPUs]={NULL};

    magma_int_t i, j, min_mn, dev, ldhpanel, lddwork, rows;
    magma_int_t ib, nb;
    magma_int_t lhwork, lwork;
    magma_int_t panel_dev, i_local, i_nb_local, n_local[MagmaMaxGPUs], la_dev, dpanel_offset;

    magma_queue_t cqueue;
    magmablasGetKernelStream( &cqueue );
    
    magma_device_t cdevice;
    magma_getdevice( &cdevice );

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    min_mn = min(m,n);
    if (min_mn == 0)
        return *info;

    nb = magma_get_zgeqrf_nb( m );

    /* dwork is (n*nb) --- for T (nb*nb) and zlarfb work ((n-nb)*nb) ---
     *        + dpanel (ldda*nb), on each GPU.
     * I think zlarfb work could be smaller, max(n_local[:]).
     * Oddly, T and zlarfb work get stacked on top of each other, both with lddwork=n.
     * on GPU that owns panel, set dpanel = dlA(dev,i,i_local).
     * on other GPUs,          set dpanel = dwork[dev] + dpanel_offset. */
    lddwork = n;
    dpanel_offset = lddwork*nb;
    for( dev=0; dev < num_gpus; dev++ ) {
        magma_setdevice( dev );
        if ( MAGMA_SUCCESS != magma_zmalloc( &(dwork[dev]), (lddwork + ldda)*nb )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto CLEANUP;
        }
    }

    /* hwork is MAX( workspace for zgeqrf (n*nb), two copies of T (2*nb*nb) )
     *        + hpanel (m*nb).
     * for last block, need 2*n*nb total. */
    ldhpanel = m;
    lhwork = max( n*nb, 2*nb*nb );
    lwork = max( lhwork + ldhpanel*nb, 2*n*nb );
    if ( MAGMA_SUCCESS != magma_zmalloc_pinned( &hwork, lwork )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto CLEANUP;
    }
    hpanel = hwork + lhwork;

    /* Set the number of local n for each GPU */
    for( dev=0; dev < num_gpus; dev++ ) {
        n_local[dev] = ((n/nb)/num_gpus)*nb;
        if (dev < (n/nb) % num_gpus)
            n_local[dev] += nb;
        else if (dev == (n/nb) % num_gpus)
            n_local[dev] += n % nb;
    }

    for( dev=0; dev < num_gpus; dev++ ) {
        magma_setdevice( dev );
        magma_queue_create( &stream[dev][0] );
        magma_queue_create( &stream[dev][1] );
        magma_event_create( &panel_event[dev] );
    }

    if ( nb < min_mn ) {
        /* Use blocked code initially */
        // Note: as written, ib cannot be < nb.
        for( i = 0; i < min_mn-nb; i += nb ) {
            /* Set the GPU number that holds the current panel */
            panel_dev = (i/nb) % num_gpus;
            
            /* Set the local index where the current panel is (j == i) */
            i_local = i/(nb*num_gpus)*nb;
            
            ib = min(min_mn-i, nb);
            rows = m-i;
            
            /* Send current panel to the CPU, after panel_event indicates it has been updated */
            magma_setdevice( panel_dev );
            magma_queue_wait_event( stream[panel_dev][1], panel_event[panel_dev] );
            magma_zgetmatrix_async( rows, ib,
                                    dlA(panel_dev, i, i_local), ldda,
                                    hpanel(i),                  ldhpanel, stream[panel_dev][1] );
            magma_queue_sync( stream[panel_dev][1] );

            // Factor panel
            lapackf77_zgeqrf( &rows, &ib, hpanel(i), &ldhpanel, tau+i,
                              hwork, &lhwork, info );
            if ( *info != 0 ) {
                fprintf( stderr, "error %d\n", (int) *info );
            }

            // Form the triangular factor of the block reflector
            // H = H(i) H(i+1) . . . H(i+ib-1)
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib,
                              hpanel(i), &ldhpanel, tau+i, hwork, &ib );

            zpanel_to_q( MagmaUpper, ib, hpanel(i), ldhpanel, hwork + ib*ib );
            // Send the current panel back to the GPUs
            for( dev=0; dev < num_gpus; dev++ ) {
                magma_setdevice( dev );
                if (dev == panel_dev)
                    dpanel[dev] = dlA(dev, i, i_local);
                else
                    dpanel[dev] = dwork[dev] + dpanel_offset;
                magma_zsetmatrix_async( rows, ib,
                                        hpanel(i),   ldhpanel,
                                        dpanel[dev], ldda, stream[dev][0] );
            }
            for( dev=0; dev < num_gpus; dev++ ) {
                magma_setdevice( dev );
                magma_queue_sync( stream[dev][0] );
            }

            // TODO: if zpanel_to_q copied whole block, wouldn't need to restore
            // -- just send the copy to the GPUs.
            // TODO: also, could zero out the lower triangle and use Azzam's larfb w/ gemm.
            
            /* Restore the panel */
            zq_to_panel( MagmaUpper, ib, hpanel(i), ldhpanel, hwork + ib*ib );

            if (i + ib < n) {
                /* Send the T matrix to the GPU. */
                for( dev=0; dev < num_gpus; dev++ ) {
                    magma_setdevice( dev );
                    magma_zsetmatrix_async( ib, ib,
                                            hwork,      ib,
                                            dwork[dev], lddwork, stream[dev][0] );
                }
                
                la_dev = (panel_dev+1) % num_gpus;
                for( dev=0; dev < num_gpus; dev++ ) {
                    magma_setdevice( dev );
                    magmablasSetKernelStream( stream[dev][0] );
                    if (dev == la_dev && i+nb < min_mn-nb) {
                        // If not last panel,
                        // for look-ahead panel, apply H' to A(i:m,i+ib:i+2*ib)
                        i_nb_local = (i+nb)/(nb*num_gpus)*nb;
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, ib, ib,
                                          dpanel[dev],             ldda,       // V
                                          dwork[dev],              lddwork,    // T
                                          dlA(dev, i, i_nb_local), ldda,       // C
                                          dwork[dev]+ib,           lddwork );  // work
                        magma_event_record( panel_event[dev], stream[dev][0] );
                        // for trailing matrix, apply H' to A(i:m,i+2*ib:n)
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, n_local[dev]-(i_nb_local+ib), ib,
                                          dpanel[dev],                ldda,       // V
                                          dwork[dev],                 lddwork,    // T
                                          dlA(dev, i, i_nb_local+ib), ldda,       // C
                                          dwork[dev]+ib,              lddwork );  // work
                    }
                    else {
                        // for trailing matrix, apply H' to A(i:m,i+ib:n)
                        i_nb_local = i_local;
                        if (dev <= panel_dev) {
                            i_nb_local += ib;
                        }
                        magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                          rows, n_local[dev]-i_nb_local, ib,
                                          dpanel[dev],             ldda,       // V
                                          dwork[dev],              lddwork,    // T
                                          dlA(dev, i, i_nb_local), ldda,       // C
                                          dwork[dev]+ib,           lddwork );  // work
                    }
                }
                // Restore top of panel (after larfb is done)
                magma_setdevice( panel_dev );
                magma_zsetmatrix_async( ib, ib,
                                        hpanel(i),                  ldhpanel,
                                        dlA(panel_dev, i, i_local), ldda, stream[panel_dev][0] );
            }
        }
    }
    else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block row. */
    if (i < min_mn) {
        rows = m-i;
        for( j=i; j < n; j += nb ) {
            panel_dev = (j/nb) % num_gpus;
            i_local = j/(nb*num_gpus)*nb;
            ib = min( n-j, nb );
            magma_setdevice( panel_dev );
            magma_zgetmatrix( rows, ib,
                              dlA(panel_dev, i, i_local), ldda,
                              hwork + (j-i)*rows,         rows );
        }

        // needs lwork >= 2*n*nb:
        // needs (m-i)*(n-i) for last block row, bounded by nb*n.
        // needs (n-i)*nb    for zgeqrf work,    bounded by n*nb.
        ib = n-i;  // total columns in block row
        lhwork = lwork - ib*rows;
        lapackf77_zgeqrf( &rows, &ib, hwork, &rows, tau+i, hwork + ib*rows, &lhwork, info );
        if ( *info != 0 ) {
            fprintf( stderr, "error %d\n", (int) *info );
        }
        
        for( j=i; j < n; j += nb ) {
            panel_dev = (j/nb) % num_gpus;
            i_local = j/(nb*num_gpus)*nb;
            ib = min( n-j, nb );
            magma_setdevice( panel_dev );
            magma_zsetmatrix( rows, ib,
                              hwork + (j-i)*rows,         rows,
                              dlA(panel_dev, i, i_local), ldda );
        }
    }

CLEANUP:
    // free(NULL) does nothing.
    // check that queues and events are non-zero before destroying them, though.
    for( dev=0; dev < num_gpus; dev++ ) {
        magma_setdevice( dev );
        if ( stream[dev][0]   ) { magma_queue_destroy( stream[dev][0]   ); }
        if ( stream[dev][1]   ) { magma_queue_destroy( stream[dev][1]   ); }
        if ( panel_event[dev] ) { magma_event_destroy( panel_event[dev] ); }
        magma_free( dwork[dev] );
    }
    magma_free_pinned( hwork );
    magma_setdevice( cdevice );
    magmablasSetKernelStream( cqueue );

    return *info;
} /* magma_zgeqrf2_mgpu */
