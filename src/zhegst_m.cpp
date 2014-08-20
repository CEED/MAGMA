/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar

       @precisions normal z -> s d c
*/
#include "common_magma.h"


static void magma_zhegst_m_1_L_col_update(magma_int_t nk, magma_int_t nb, magmaDoubleComplex* dA_col, magma_int_t ldda,
                                          magmaDoubleComplex* dC1, magma_int_t lddc1, magmaDoubleComplex* dC2, magma_int_t lddc2);

static void magma_zhegst_m_1_U_row_update(magma_int_t nk, magma_int_t nb, magmaDoubleComplex* dA_row, magma_int_t ldda,
                                          magmaDoubleComplex* dC1, magma_int_t lddc1, magmaDoubleComplex* dC2, magma_int_t lddc2);

/**
    Purpose
    -------
    ZHEGST_M reduces a complex Hermitian-definite generalized
    eigenproblem to standard form.
    
    If ITYPE = 1, the problem is A*x = lambda*B*x,
    and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
    
    If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
    B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
    
    B must have been previously factorized as U**H*U or L*L**H by ZPOTRF.
    
    Arguments
    ---------
    @param[in]
    nrgpu   INTEGER
            Number of GPUs to use.

    @param[in]
    itype   INTEGER
            = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
            = 2 or 3: compute U*A*U**H or L**H*A*L.
    
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored and B is factored as U**H*U;
      -     = MagmaLower:  Lower triangle of A is stored and B is factored as L*L**H.
    
    @param[in]
    n       INTEGER
            The order of the matrices A and B.  N >= 0.
    
    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the transformed matrix, stored in the
            same format as A.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).
    
    @param[in]
    B       COMPLEX_16 array, dimension (LDB,N)
            The triangular factor from the Cholesky factorization of B,
            as returned by ZPOTRF.
    
    @param[in]
    ldb     INTEGER
            The leading dimension of the array B.  LDB >= max(1,N).
    
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
    

    @ingroup magma_zheev_comp
    ********************************************************************/
extern "C" magma_int_t
magma_zhegst_m(magma_int_t nrgpu, magma_int_t itype, magma_uplo_t uplo, magma_int_t n,
               magmaDoubleComplex *A, magma_int_t lda,
               magmaDoubleComplex *B, magma_int_t ldb, magma_int_t *info)
{
#define A(i, j) (A + (j)*nb*lda + (i)*nb)
#define B(i, j) (B + (j)*nb*ldb + (i)*nb)
#define dA(gpui, i, j)    (dw[gpui] + (j)*nb*ldda + (i)*nb)
#define dB_c(gpui, i, j)  (dw[gpui] + dima*ldda + (i)*nb + (j)*nb*lddbc)
#define dB_r(gpui, i, j)  (dw[gpui] + dima*ldda + (i)*nb + (j)*nb*lddbr)
#define dwork(gpui, i, j) (dw[gpui] + dima*ldda + lddbc*lddbr + (j)*nb*nb + (i)*nb)

    const char* uplo_ = lapack_uplo_const( uplo );

    double             d_one      = 1.0;
    magmaDoubleComplex    c_one      = MAGMA_Z_ONE;
    magmaDoubleComplex    c_neg_one  = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex    c_half     = MAGMA_Z_HALF;
    magmaDoubleComplex    c_neg_half = MAGMA_Z_NEG_HALF;
    magmaDoubleComplex* dw[MagmaMaxGPUs];

    magma_queue_t stream [MagmaMaxGPUs][3];
    magma_event_t  event  [MagmaMaxGPUs][2];

    int upper = (uplo == MagmaUpper);

    magma_int_t nb = magma_get_zhegst_nb_m(n);

    /* Test the input parameters. */
    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (! upper && uplo != MagmaLower) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < max(1,n)) {
        *info = -5;
    } else if (ldb < max(1,n)) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return */
    if ( n == 0 )
        return *info;

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );

    magma_int_t nbl = (n-1)/nb+1; // number of blocks

    magma_int_t ldda = 0;
    magma_int_t dima = 0;

    if ( (itype == 1 && upper) || (itype != 1 && !upper) ) {
        ldda = ((nbl-1)/nrgpu+1)*nb;
        dima = n;
    } else {
        ldda = n;
        dima = ((nbl-1)/nrgpu+1)*nb;
    }
    magma_int_t lddbr = 2 * nb;
    magma_int_t lddbc = n;

    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
        magma_setdevice(igpu);

        if (MAGMA_SUCCESS != magma_zmalloc( &dw[igpu], (dima*ldda + lddbc*lddbr + n*nb) )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            return *info;
        }
        magma_queue_create( &stream[igpu][0] );
        magma_queue_create( &stream[igpu][1] );
        magma_queue_create( &stream[igpu][2] );
        magma_event_create( &event[igpu][0] );
        magma_event_create( &event[igpu][1] );
    }

    /* Use hybrid blocked code */

    if (itype == 1) {
        if (upper) {
            /* Compute inv(U')*A*inv(U) */

            //copy A to mgpu
            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t igpu = k%nrgpu;
                magma_setdevice(igpu);
                magma_int_t kb = min(nb, n-k*nb);
                magma_zsetmatrix_async(kb, n-k*nb,
                                       A(k, k),              lda,
                                       dA(igpu, k/nrgpu, k), ldda, stream[igpu][0] );
            }

            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t ind_k  =   k   % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min(n-k*nb,nb);

                // Copy B panel
                for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                    magma_setdevice(igpu);
                    magma_queue_sync( stream[igpu][0] ); // sync previous B panel copy

                    // sync dwork copy and update (uses B panel of the next copy)
                    magma_queue_wait_event( stream[igpu][0], event[igpu][1] );
                    magma_queue_wait_event( stream[igpu][2], event[igpu][1] );
                    magma_zsetmatrix_async(kb, n-k*nb,
                                           B(k, k),              ldb,
                                           dB_r(igpu, ind_k, k), lddbr, stream[igpu][0] );
                }

                magma_int_t igpu_p = k%nrgpu;

                if (k > 0) {
                    // Update the next panel
                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_int_t nk = n - k*nb;

                    magma_zher2k(MagmaUpper, MagmaConjTrans, kb, nb,
                                 c_neg_one, dwork(igpu_p, 0, k), nb, dB_r(igpu_p, ind_k1, k), lddbr,
                                 d_one, dA(igpu_p, k/nrgpu, k), ldda);

                    // copy Akk block on the CPU
                    magma_zgetmatrix_async(kb, kb,
                                           dA(igpu_p, k/nrgpu, k), ldda,
                                           A(k, k),                lda, stream[igpu_p][2] );

                    magma_event_record( event[igpu_p][0], stream[igpu_p][2]);

                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb, c_neg_one, dwork(igpu_p, 0, k), nb,
                                dB_r(igpu_p, ind_k1, k+1), lddbr, c_one, dA(igpu_p, k/nrgpu, k+1), ldda );

                    magma_zgemm(MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb, c_neg_one, dB_r(igpu_p, ind_k1, k), lddbr,
                                dwork(igpu_p, 0, k+1), nb, c_one, dA(igpu_p, k/nrgpu, k+1), ldda );

                    // Update the panels of the other GPUs
                    for (magma_int_t j=k+1; j < nbl; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        if (igpu != igpu_p) {
                            magma_setdevice(igpu);
                            magmablasSetKernelStream(stream[igpu][1]);

                            magma_zhegst_m_1_U_row_update(n-j*nb, nb, dA(igpu, j/nrgpu, j), ldda,
                                                          dwork(igpu, 0, j), nb, dB_r(igpu, ind_k1, j), lddbr); // zher2k on j-th row
                        }
                    }
                }
                // compute next panel
                magma_setdevice(igpu_p);

                if (k+1 < nbl) {
                    magma_queue_sync( stream[igpu_p][0] ); // sync B panel copy
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_ztrsm(MagmaLeft, uplo, MagmaConjTrans, MagmaNonUnit,
                                kb, n-(k+1)*nb,
                                c_one, dB_r(igpu_p, ind_k, k), lddbr,
                                dA(igpu_p, k/nrgpu, k+1), ldda);
                }

                magma_event_sync( event[igpu_p][0] ); // sync Akk copy
                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                if (k+1 < nbl) {
                    magma_zsetmatrix_async(kb, kb,
                                           A(k, k),              lda,
                                           dA(igpu_p, k/nrgpu, k), ldda, stream[igpu_p][2] );

                    magma_zhemm(MagmaLeft, uplo,
                                kb, n-(k+1)*nb,
                                c_neg_half, dA(igpu_p, k/nrgpu, k), ldda,
                                dB_r(igpu_p, ind_k, k+1), lddbr,
                                c_one, dA(igpu_p, k/nrgpu, k+1), ldda);

                    magma_zgetmatrix_async(kb, n-(k+1)*nb,
                                           dA(igpu_p, k/nrgpu, k+1), ldda,
                                           A(k, k+1),                lda, stream[igpu_p][2] );
                }

                if (k > 0) {
                    // Update the remaining panels of GPU igpu_p
                    for (magma_int_t j=k+nrgpu; j < nbl; j += nrgpu) {
                        magma_setdevice(igpu_p);
                        magmablasSetKernelStream(stream[igpu_p][1]);

                        magma_zhegst_m_1_U_row_update(n-j*nb, nb, dA(igpu_p, j/nrgpu, j), ldda,
                                                      dwork(igpu_p, 0, j), nb, dB_r(igpu_p, ind_k1, j), lddbr); // zher2k on j-th row
                    }
                }

                if (k+1 < nbl) {
                    // send the partially updated panel of dA to each gpu in dwork block
                    magma_setdevice(igpu_p);
                    magma_queue_sync( stream[igpu_p][2] );

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        magma_zsetmatrix_async(kb, n-(k+1)*nb,
                                               A(k, k+1),          lda,
                                               dwork(igpu, 0, k+1), nb, stream[igpu][1] );

                        magma_event_record( event[igpu][1], stream[igpu][1]);
                    }

                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][1]);

                    magma_zhemm(MagmaLeft, uplo,
                                kb, n-(k+1)*nb,
                                c_neg_half, dA(igpu_p, k/nrgpu, k), ldda,
                                dB_r(igpu_p, ind_k, k+1), lddbr,
                                c_one, dA(igpu_p, k/nrgpu, k+1), ldda);
                }
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][1] );
            }

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 1; j < nbl; ++j) {
                    nloc[(j-1)%nrgpu] += nb;

                    magma_int_t jb = min(nb, n-j*nb);

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        if (nloc[igpu] > 0) {
                            magma_zsetmatrix_async(jb, n-j*nb,
                                                   B(j, j),            ldb,
                                                   dB_r(igpu, j%2, j), lddbr, stream[igpu][j%2] );

                            magma_queue_wait_event( stream[igpu][j%2], event[igpu][0] );

                            magmablasSetKernelStream(stream[igpu][j%2]);
                            magma_ztrsm(MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit, nloc[igpu], jb, c_one, dB_r(igpu, j%2, j), lddbr,
                                        dA(igpu, 0, j), ldda );

                            if ( j < nbl-1 ) {
                                magma_zgemm(MagmaNoTrans, MagmaNoTrans, nloc[igpu], n-(j+1)*nb, nb, c_neg_one, dA(igpu, 0, j), ldda,
                                            dB_r(igpu, j%2, j+1), lddbr, c_one, dA(igpu, 0, j+1), ldda );
                            }
                            magma_event_record( event[igpu][0], stream[igpu][j%2]);
                        }
                    }

                    for (magma_int_t k = 0; k < j; ++k) {
                        magma_int_t igpu = k%nrgpu;
                        magma_setdevice(igpu);
                        magma_int_t kb = min(nb, n-k*nb);
                        magma_zgetmatrix_async(kb, jb,
                                               dA(igpu, k/nrgpu, j), ldda,
                                               A(k, j),              lda, stream[igpu][j%2] );
                    }
                }
            }
        } else {
            /* Compute inv(L)*A*inv(L') */

            // Copy A to mgpu
            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t igpu = k%nrgpu;
                magma_setdevice(igpu);
                magma_int_t kb = min(nb, n-k*nb);
                magma_zsetmatrix_async((n-k*nb), kb,
                                       A(k, k),              lda,
                                       dA(igpu, k, k/nrgpu), ldda, stream[igpu][0] );
            }

            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t ind_k  =   k   % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min(n-k*nb,nb);

                // Copy B panel
                for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                    magma_setdevice(igpu);
                    magma_queue_sync( stream[igpu][0] ); // sync previous B panel copy

                    // sync dwork copy and update (uses B panel of the next copy)
                    magma_queue_wait_event( stream[igpu][0], event[igpu][1] );
                    magma_queue_wait_event( stream[igpu][2], event[igpu][1] );
                    magma_zsetmatrix_async((n-k*nb), kb,
                                           B(k, k),          ldb,
                                           dB_c(igpu, k, ind_k), lddbc, stream[igpu][0] );
                }

                magma_int_t igpu_p = k%nrgpu;

                if (k > 0) {
                    // Update the next panel
                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_int_t nk = n-k*nb;

                    magma_zher2k(MagmaLower, MagmaNoTrans, kb, nb,
                                 c_neg_one, dwork(igpu_p, k, 0), n, dB_c(igpu_p, k, ind_k1), lddbc,
                                 d_one, dA(igpu_p, k, k/nrgpu), ldda);

                    // copy Akk block on the CPU
                    magma_zgetmatrix_async(kb, kb,
                                           dA(igpu_p, k, k/nrgpu), ldda,
                                           A(k, k),                lda, stream[igpu_p][2] );

                    magma_event_record( event[igpu_p][0], stream[igpu_p][2]);

                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb, c_neg_one, dwork(igpu_p, k+1, 0), n,
                                dB_c(igpu_p, k, ind_k1), lddbc, c_one, dA(igpu_p, k+1, k/nrgpu), ldda );

                    magma_zgemm(MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb, c_neg_one, dB_c(igpu_p, k+1, ind_k1), lddbc,
                                dwork(igpu_p, k, 0), n, c_one, dA(igpu_p, k+1, k/nrgpu), ldda );

                    // Update the panels of the other GPUs
                    for (magma_int_t j=k+1; j < nbl; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        if (igpu != igpu_p) {
                            magma_setdevice(igpu);
                            magmablasSetKernelStream(stream[igpu][1]);

                            magma_zhegst_m_1_L_col_update(n-j*nb, nb, dA(igpu, j, j/nrgpu), ldda,
                                                          dwork(igpu, j, 0), n, dB_c(igpu, j, ind_k1), lddbc); // zher2k on j-th column
                        }
                    }
                }
                // compute next panel
                magma_setdevice(igpu_p);

                if (k+1 < nbl) {
                    magma_queue_sync( stream[igpu_p][0] ); // sync B panel copy
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_ztrsm(MagmaRight, uplo, MagmaConjTrans, MagmaNonUnit,
                                n-(k+1)*nb, kb,
                                c_one, dB_c(igpu_p, k, ind_k), lddbc,
                                dA(igpu_p, k+1, k/nrgpu), ldda);
                }

                magma_event_sync( event[igpu_p][0] ); // sync Akk copy
                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                if (k+1 < nbl) {
                    magma_zsetmatrix_async(kb, kb,
                                           A(k, k),                lda,
                                           dA(igpu_p, k, k/nrgpu), ldda, stream[igpu_p][2] );

                    magma_zhemm(MagmaRight, uplo,
                                n-(k+1)*nb, kb,
                                c_neg_half, dA(igpu_p, k, k/nrgpu), ldda,
                                dB_c(igpu_p, k+1, ind_k), lddbc,
                                c_one, dA(igpu_p, k+1, k/nrgpu), ldda);

                    magma_zgetmatrix_async(n-(k+1)*nb, kb,
                                           dA(igpu_p, k+1, k/nrgpu), ldda,
                                           A(k+1, k),                lda, stream[igpu_p][2] );
                }

                if (k > 0) {
                    // Update the remaining panels of GPU igpu_p
                    for (magma_int_t j=k+nrgpu; j < nbl; j += nrgpu) {
                        magma_setdevice(igpu_p);
                        magmablasSetKernelStream(stream[igpu_p][1]);

                        magma_zhegst_m_1_L_col_update(n-j*nb, nb, dA(igpu_p, j, j/nrgpu), ldda,
                                                      dwork(igpu_p, j, 0), n, dB_c(igpu_p, j, ind_k1), lddbc); // zher2k on j-th column
                    }
                }

                if (k+1 < nbl) {
                    // send the partially updated panel of dA to each gpu in dwork block
                    magma_setdevice(igpu_p);
                    magma_queue_sync( stream[igpu_p][2] );

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        magma_zsetmatrix_async((n-(k+1)*nb), kb,
                                               A(k+1, k),          lda,
                                               dwork(igpu, k+1, 0), n, stream[igpu][1] );

                        magma_event_record( event[igpu][1], stream[igpu][1]);
                    }

                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][1]);

                    magma_zhemm(MagmaRight, uplo,
                                n-(k+1)*nb, kb,
                                c_neg_half, dA(igpu_p, k, k/nrgpu), ldda,
                                dB_c(igpu_p, k+1, ind_k), lddbc,
                                c_one, dA(igpu_p, k+1, k/nrgpu), ldda);
                }
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][1] );
            }

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 1; j < nbl; ++j) {
                    nloc[(j-1)%nrgpu] += nb;

                    magma_int_t jb = min(nb, n-j*nb);

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        if (nloc[igpu] > 0) {
                            magma_zsetmatrix_async((n-j*nb), jb,
                                                   B(j, j),            ldb,
                                                   dB_c(igpu, j, j%2), lddbc, stream[igpu][j%2] );

                            magma_queue_wait_event( stream[igpu][j%2], event[igpu][0] );

                            magmablasSetKernelStream(stream[igpu][j%2]);
                            magma_ztrsm(MagmaLeft, uplo, MagmaNoTrans, MagmaNonUnit, jb, nloc[igpu], c_one, dB_c(igpu, j, j%2), lddbc,
                                        dA(igpu, j, 0), ldda );

                            if ( j < nbl-1 ) {
                                magma_zgemm(MagmaNoTrans, MagmaNoTrans, n-(j+1)*nb, nloc[igpu], nb, c_neg_one, dB_c(igpu, j+1, j%2), lddbc,
                                            dA(igpu, j, 0), ldda, c_one, dA(igpu, j+1, 0), ldda );
                            }
                            magma_event_record( event[igpu][0], stream[igpu][j%2]);
                        }
                    }

                    for (magma_int_t k = 0; k < j; ++k) {
                        magma_int_t igpu = k%nrgpu;
                        magma_setdevice(igpu);
                        magma_int_t kb = min(nb, n-k*nb);
                        magma_zgetmatrix_async(jb, kb,
                                               dA(igpu, j, k/nrgpu), ldda,
                                               A(j, k),              lda, stream[igpu][j%2] );
                    }
                }
            }
        }
    } else {
        if (upper) {
            /* Compute U*A*U' */

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };
                magma_int_t iloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 0; j < nbl; ++j) {
                    magma_int_t jb = min(nb, n-j*nb);
                    nloc[j%nrgpu] += jb;
                }

                for (magma_int_t k = 0; k < nbl; ++k) {
                    magma_int_t kb = min(nb, n-k*nb);

                    for (magma_int_t j = k; j < nbl; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        magma_setdevice(igpu);
                        magma_int_t jb = min(nb, n-j*nb);
                        magma_zsetmatrix_async(kb, jb,
                                               A(k, j),              lda,
                                               dA(igpu, k, j/nrgpu), ldda, stream[igpu][k%2] );
                    }

                    magma_int_t igpu_p = k % nrgpu;

                    ++iloc[igpu_p];
                    nloc[igpu_p] -= kb;

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        magma_zsetmatrix_async(k*nb + kb, kb,
                                               B(0, k),            ldb,
                                               dB_c(igpu, 0, k%2), lddbc, stream[igpu][k%2] );

                        magma_queue_wait_event( stream[igpu][k%2], event[igpu][0] );

                        magmablasSetKernelStream(stream[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zhemm(MagmaRight, uplo,
                                        k*nb, kb,
                                        c_half, dA(igpu, k, k/nrgpu), ldda,
                                        dB_c(igpu, 0, k%2), lddbc,
                                        c_one, dA(igpu, 0, k/nrgpu), ldda);
                        }

                        magma_zgemm(MagmaNoTrans, MagmaNoTrans, k*nb, nloc[igpu], kb, c_one, dB_c(igpu, 0, k%2), lddbc,
                                    dA(igpu, k, iloc[igpu]), ldda, c_one, dA(igpu, 0, iloc[igpu]), ldda );

                        magma_ztrmm(MagmaLeft, uplo, MagmaNoTrans, MagmaNonUnit, kb, nloc[igpu], c_one, dB_c(igpu, k, k%2), lddbc,
                                    dA(igpu, k, iloc[igpu]), ldda );

                        magma_event_record( event[igpu][0], stream[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zgetmatrix_async(k*nb, kb,
                                                   dA(igpu, 0, k/nrgpu), ldda,
                                                   A(0, k),              lda, stream[igpu][k%2] );
                        }
                    }
                }
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][0] );
                magma_queue_sync( stream[igpu][1] );
            }

            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t ind_k = k % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min(n-k*nb,nb);

                magma_int_t igpu_p = k%nrgpu;

                if (k > 0) {
                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);

                        magma_queue_wait_event( stream[igpu][0], event[igpu][ind_k] ); // sync computation that use the B panel of next copy

                        magma_zsetmatrix_async(k*nb+kb, kb,
                                               B(0, k),              ldb,
                                               dB_c(igpu, 0, ind_k), lddbc, stream[igpu][0] );

                        magma_event_record(event[igpu][ind_k], stream[igpu][0]);

                        magma_zsetmatrix_async(k*nb, kb,
                                               A(0, k),           lda,
                                               dwork(igpu, 0, 0), n, stream[igpu][1] );

                        magma_queue_wait_event( stream[igpu][1], event[igpu][ind_k] ); // sync B copy
                    }

                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_queue_wait_event( stream[igpu_p][2], event[igpu_p][ind_k1] ); // sync update of previous step
                    magma_queue_wait_event( stream[igpu_p][2], event[igpu_p][ind_k] ); // sync B copy

                    magma_zhemm(MagmaRight, uplo,
                                k*nb, kb,
                                c_half, dA(igpu_p, k, k/nrgpu), ldda,
                                dB_c(igpu_p, 0, ind_k), lddbc,
                                c_one, dA(igpu_p, 0, k/nrgpu), ldda);

                    magma_ztrmm(MagmaRight, uplo, MagmaConjTrans, MagmaNonUnit,
                                k*nb, kb,
                                c_one, dB_c(igpu_p, k, ind_k), lddbc,
                                dA(igpu_p, 0, k/nrgpu), ldda);

                    magma_event_record(event[igpu_p][ind_k], stream[igpu_p][2]);
                    magma_queue_wait_event(stream[igpu_p][1], event[igpu_p][ind_k]);

                    for (magma_int_t j = 0; j < k; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        magma_setdevice(igpu);
                        magmablasSetKernelStream(stream[igpu][1]);

                        magma_zher2k(uplo, MagmaNoTrans,
                                     nb, kb,
                                     c_one, dwork(igpu, j, 0), n,
                                     dB_c(igpu, j, ind_k), lddbc,
                                     d_one, dA(igpu, j, j/nrgpu), ldda);

                        magma_zgemm(MagmaNoTrans, MagmaConjTrans, j*nb, nb, kb, c_one, dB_c(igpu, 0, ind_k), lddbc,
                                    dwork(igpu, j, 0), n, c_one, dA(igpu, 0, j/nrgpu), ldda );

                        magma_zgemm(MagmaNoTrans, MagmaConjTrans, j*nb, nb, kb, c_one, dwork(igpu, 0, 0), n,
                                    dB_c(igpu, j, ind_k), lddbc, c_one, dA(igpu, 0, j/nrgpu), ldda );
                    }
                }

                for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                    magma_setdevice(igpu);
                    magma_event_record(event[igpu][ind_k], stream[igpu][1]);
                    magma_queue_sync(stream[igpu][0]); // sync B copy (conflicts with zhegst)
                }

                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                magma_setdevice(igpu_p);

                magma_zsetmatrix_async(kb, kb,
                                       A(k, k),                lda,
                                       dA(igpu_p, k, k/nrgpu), ldda, stream[igpu_p][1] );
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][1] );
            }

            //copy A from mgpus
            for (magma_int_t j = 0; j < nbl; ++j) {
                magma_int_t igpu = j%nrgpu;
                magma_setdevice(igpu);
                magma_int_t jb = min(nb, n-j*nb);
                magma_zgetmatrix_async(j*nb+jb, jb,
                                       dA(igpu, 0, j/nrgpu), ldda,
                                       A(0, j),              lda, stream[igpu][0] );
            }
        } else {
            /* Compute L'*A*L */

            if (n > nb) {
                magma_int_t nloc[MagmaMaxGPUs] = { 0 };
                magma_int_t iloc[MagmaMaxGPUs] = { 0 };

                for (magma_int_t j = 0; j < nbl; ++j) {
                    magma_int_t jb = min(nb, n-j*nb);
                    nloc[j%nrgpu] += jb;
                }

                for (magma_int_t k = 0; k < nbl; ++k) {
                    magma_int_t kb = min(nb, n-k*nb);

                    for (magma_int_t j = k; j < nbl; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        magma_setdevice(igpu);
                        magma_int_t jb = min(nb, n-j*nb);

                        magma_zsetmatrix_async(jb, kb,
                                               A(j, k),              lda,
                                               dA(igpu, j/nrgpu, k), ldda, stream[igpu][k%2] );
                    }

                    magma_int_t igpu_p = k % nrgpu;

                    ++iloc[igpu_p];
                    nloc[igpu_p] -= kb;

                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);
                        magma_zsetmatrix_async(kb, k*nb +kb,
                                               B(k, 0),            ldb,
                                               dB_r(igpu, k%2, 0), lddbr, stream[igpu][k%2] );

                        magma_queue_wait_event( stream[igpu][k%2], event[igpu][0] );

                        magmablasSetKernelStream(stream[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zhemm(MagmaLeft, uplo,
                                        kb, k*nb,
                                        c_half, dA(igpu, k/nrgpu, k), ldda,
                                        dB_r(igpu, k%2, 0), lddbr,
                                        c_one, dA(igpu, k/nrgpu, 0), ldda);
                        }

                        magma_zgemm(MagmaNoTrans, MagmaNoTrans, nloc[igpu], k*nb, kb, c_one, dA(igpu, iloc[igpu], k), ldda,
                                    dB_r(igpu, k%2, 0), lddbr, c_one, dA(igpu, iloc[igpu], 0), ldda );

                        magma_ztrmm(MagmaRight, uplo, MagmaNoTrans, MagmaNonUnit, nloc[igpu], kb, c_one, dB_r(igpu, k%2, k), lddbr,
                                    dA(igpu, iloc[igpu], k), ldda );

                        magma_event_record( event[igpu][0], stream[igpu][k%2]);

                        if (igpu == igpu_p) {
                            magma_zgetmatrix_async(kb, k*nb,
                                                   dA(igpu, k/nrgpu, 0), ldda,
                                                   A(k, 0),              lda, stream[igpu][k%2] );
                        }
                    }
                }
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][0] );
                magma_queue_sync( stream[igpu][1] );
            }

            for (magma_int_t k = 0; k < nbl; ++k) {
                magma_int_t ind_k = k % 2;
                magma_int_t ind_k1 = (k+1) % 2;
                magma_int_t kb= min(n-k*nb,nb);

                magma_int_t igpu_p = k%nrgpu;

                if (k > 0) {
                    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                        magma_setdevice(igpu);

                        magma_queue_wait_event( stream[igpu][0], event[igpu][ind_k] ); // sync computation that use the B panel of next copy

                        magma_zsetmatrix_async(kb, k*nb+kb,
                                               B(k, 0),              ldb,
                                               dB_r(igpu, ind_k, 0), lddbr, stream[igpu][0] );

                        magma_event_record(event[igpu][ind_k], stream[igpu][0]);

                        magma_zsetmatrix_async(kb, k*nb,
                                               A(k, 0),           lda,
                                               dwork(igpu, 0, 0), nb, stream[igpu][1] );

                        magma_queue_wait_event( stream[igpu][1], event[igpu][ind_k] ); // sync B copy
                    }

                    magma_setdevice(igpu_p);
                    magmablasSetKernelStream(stream[igpu_p][2]);

                    magma_queue_wait_event( stream[igpu_p][2], event[igpu_p][ind_k1] ); // sync update of previous step
                    magma_queue_wait_event( stream[igpu_p][2], event[igpu_p][ind_k] ); // sync B copy

                    magma_zhemm(MagmaLeft, uplo,
                                kb, k*nb,
                                c_half, dA(igpu_p, k/nrgpu, k), ldda,
                                dB_r(igpu_p, ind_k, 0), lddbr,
                                c_one, dA(igpu_p, k/nrgpu, 0), ldda);

                    magma_ztrmm(MagmaLeft, uplo, MagmaConjTrans, MagmaNonUnit,
                                kb, k*nb,
                                c_one, dB_r(igpu_p, ind_k, k), lddbr,
                                dA(igpu_p, k/nrgpu, 0), ldda);

                    magma_event_record(event[igpu_p][ind_k], stream[igpu_p][2]);
                    magma_queue_wait_event(stream[igpu_p][1], event[igpu_p][ind_k]);

                    for (magma_int_t j = 0; j < k; ++j) {
                        magma_int_t igpu = j%nrgpu;
                        magma_setdevice(igpu);
                        magmablasSetKernelStream(stream[igpu][1]);

                        magma_zher2k(uplo, MagmaConjTrans,
                                     nb, kb,
                                     c_one, dwork(igpu, 0, j), nb,
                                     dB_r(igpu, ind_k, j), lddbr,
                                     d_one, dA(igpu, j/nrgpu, j), ldda);

                        magma_zgemm(MagmaConjTrans, MagmaNoTrans, nb, j*nb, kb, c_one, dwork(igpu, 0, j), nb,
                                    dB_r(igpu, ind_k, 0), lddbr, c_one, dA(igpu, j/nrgpu, 0), ldda );

                        magma_zgemm(MagmaConjTrans, MagmaNoTrans, nb, j*nb, kb, c_one, dB_r(igpu, ind_k, j), lddbr,
                                    dwork(igpu, 0, 0), nb, c_one, dA(igpu, j/nrgpu, 0), ldda );
                    }
                }

                for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                    magma_setdevice(igpu);
                    magma_event_record(event[igpu][ind_k], stream[igpu][1]);
                    magma_queue_sync(stream[igpu][0]); // sync B copy (conflicts with zhegst)
                }

                lapackf77_zhegst( &itype, uplo_, &kb, A(k,k), &lda, B(k,k), &ldb, info);

                magma_setdevice(igpu_p);

                magma_zsetmatrix_async(kb, kb,
                                       A(k, k),                lda,
                                       dA(igpu_p, k/nrgpu, k), ldda, stream[igpu_p][1] );
            }

            for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
                magma_queue_sync( stream[igpu][1] );
            }

            //copy A from mgpus
            for (magma_int_t j = 0; j < nbl; ++j) {
                magma_int_t igpu = j%nrgpu;
                magma_setdevice(igpu);
                magma_int_t jb = min(nb, n-j*nb);
                magma_zgetmatrix_async(jb, j*nb+jb,
                                       dA(igpu, j/nrgpu, 0), ldda,
                                       A(j, 0),              lda, stream[igpu][0] );
            }
        }
    }

    for (magma_int_t igpu = 0; igpu < nrgpu; ++igpu) {
        magma_setdevice(igpu);
        magma_event_destroy( event[igpu][0] );
        magma_event_destroy( event[igpu][1] );
        magma_queue_destroy( stream[igpu][0] );
        magma_queue_destroy( stream[igpu][1] );
        magma_queue_destroy( stream[igpu][2] );
        magma_free( dw[igpu] );
    }
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_zhegst_gpu */


inline static void magma_zhegst_m_1_U_row_update(
    magma_int_t nk, magma_int_t nb,
    magmaDoubleComplex* dA_row, magma_int_t ldda,
    magmaDoubleComplex* dC1, magma_int_t lddc1,
    magmaDoubleComplex* dC2, magma_int_t lddc2)
{
    // update 1 rowblock (rowwise zher2k) for itype=1 Upper case
    double             d_one      = 1.0;
    magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;

    magma_int_t kb = min(nk, nb);

    magma_zher2k(MagmaUpper, MagmaConjTrans, kb, nb,
                 c_neg_one, dC1, lddc1, dC2, lddc2,
                 d_one, dA_row, ldda);

    magma_zgemm(MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb, c_neg_one, dC1, lddc1,
                dC2+kb*lddc2, lddc2, c_one, dA_row+kb*ldda, ldda );

    magma_zgemm(MagmaConjTrans, MagmaNoTrans, kb, nk-kb, nb, c_neg_one, dC2, lddc2,
                dC1+kb*lddc1, lddc1, c_one, dA_row+kb*ldda, ldda );
}


inline static void magma_zhegst_m_1_L_col_update(
    magma_int_t nk, magma_int_t nb,
    magmaDoubleComplex* dA_col, magma_int_t ldda,
    magmaDoubleComplex* dC1, magma_int_t lddc1,
    magmaDoubleComplex* dC2, magma_int_t lddc2)
{
    // update 1 columnblock (columnwise zher2k) for itype=1 Lower case
    double             d_one      = 1.0;
    magmaDoubleComplex c_one      = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one  = MAGMA_Z_NEG_ONE;

    magma_int_t kb = min(nk, nb);

    magma_zher2k(MagmaLower, MagmaNoTrans, kb, nb,
                 c_neg_one, dC1, lddc1, dC2, lddc2,
                 d_one, dA_col, ldda);

    magma_zgemm(MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb, c_neg_one, dC1+kb, lddc1,
                dC2, lddc2, c_one, dA_col+kb, ldda );

    magma_zgemm(MagmaNoTrans, MagmaConjTrans, nk-kb, kb, nb, c_neg_one, dC2+kb, lddc2,
                dC1, lddc1, c_one, dA_col+kb, ldda );
}
