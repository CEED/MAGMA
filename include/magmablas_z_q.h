/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
*/

#ifndef MAGMABLAS_Z_Q_H
#define MAGMABLAS_Z_Q_H
                    
#include "magma_types.h"
#include "magma_copy_q.h"

#define COMPLEX

#ifdef __cplusplus
extern "C" {
#endif

  /*
   * Transpose functions
   */
void
magmablas_ztranspose_inplace_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ztranspose_conj_inplace_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_ztranspose_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_ztranspose_conj_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA,  magma_int_t ldda,
    magmaDoubleComplex_ptr       dAT, magma_int_t lddat,
    magma_queue_t queue );

void
magmablas_zgetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dAT,   magma_int_t ldda,
    magmaDoubleComplex          *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr       dwork, magma_int_t lddwork, magma_int_t nb,
    magma_queue_t queues[2] );

void
magmablas_zsetmatrix_transpose_q(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,    magma_int_t lda,
    magmaDoubleComplex_ptr    dAT,   magma_int_t ldda,
    magmaDoubleComplex_ptr    dwork, magma_int_t lddwork, magma_int_t nb,
    magma_queue_t queues[2] );

  /*
   * RBT-related functions
   */
void
magmablas_zprbt_q(
    magma_int_t n, 
    magmaDoubleComplex_ptr dA, magma_int_t ldda, 
    magmaDoubleComplex_ptr du,
    magmaDoubleComplex_ptr dv,
    magma_queue_t queue );

void
magmablas_zprbt_mv_q(
    magma_int_t n, 
    magmaDoubleComplex_ptr dv,
    magmaDoubleComplex_ptr db,
    magma_queue_t queue );

void
magmablas_zprbt_mtv_q(
    magma_int_t n, 
    magmaDoubleComplex_ptr du,
    magmaDoubleComplex_ptr db,
    magma_queue_t queue );

  /*
   * Multi-GPU copy functions
   */
void
magma_zgetmatrix_1D_col_bcyclic_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda,
    magmaDoubleComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_zsetmatrix_1D_col_bcyclic_q(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_zgetmatrix_1D_row_bcyclic_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr const dA[], magma_int_t ldda,
    magmaDoubleComplex                *hA,   magma_int_t lda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magma_zsetmatrix_1D_row_bcyclic_q(
    magma_int_t m, magma_int_t n,
    const magmaDoubleComplex *hA,   magma_int_t lda,
    magmaDoubleComplex_ptr    dA[], magma_int_t ldda,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queue[ MagmaMaxGPUs ] );

void
magmablas_zgetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    magmaDoubleComplex_const_ptr const dAT[],    magma_int_t ldda,
    magmaDoubleComplex                *hA,       magma_int_t lda,
    magmaDoubleComplex_ptr             dwork[],  magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

void
magmablas_zsetmatrix_transpose_mgpu(
    magma_int_t ngpu, magma_queue_t queues[][2],
    const magmaDoubleComplex *hA,      magma_int_t lda,
    magmaDoubleComplex_ptr    dAT[],   magma_int_t ldda,
    magmaDoubleComplex_ptr    dwork[], magma_int_t lddw,
    magma_int_t m, magma_int_t n, magma_int_t nb );

// in src/zhetrd_mgpu.cpp
// TODO rename zsetmatrix_sy or similar
magma_int_t
magma_zhtodhe(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t n, magma_int_t nb,
    magmaDoubleComplex     *A,   magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][10],
    magma_int_t *info );

// in src/zpotrf3_mgpu.cpp
// TODO same as magma_zhtodhe?
magma_int_t
magma_zhtodpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb,
    magmaDoubleComplex     *A,   magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );

// in src/zpotrf3_mgpu.cpp
// TODO rename zgetmatrix_sy or similar
magma_int_t
magma_zdtohpo(
    magma_int_t ngpu, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t off_i, magma_int_t off_j, magma_int_t nb, magma_int_t NB,
    magmaDoubleComplex     *A,   magma_int_t lda,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda,
    magma_queue_t queues[][3],
    magma_int_t *info );


  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
  /*
   * Multi-GPU BLAS functions (alphabetical order)
   */
void
magmablas_zhemm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t dworksiz,
    magmaDoubleComplex    *C,       magma_int_t ldc,
    magmaDoubleComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][20], magma_int_t nbevents );

void
magmablas_zhemm_mgpu_com(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t dworksiz,
    magmaDoubleComplex    *C,       magma_int_t ldc,
    magmaDoubleComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

void
magmablas_zhemm_mgpu_spec(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t dworksiz,
    magmaDoubleComplex    *C,       magma_int_t ldc,
    magmaDoubleComplex    *work[],  magma_int_t worksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t redevents[][MagmaMaxGPUs*MagmaMaxGPUs+10], magma_int_t nbevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs+2], magma_int_t nbcmplx );

magma_int_t
magmablas_zhemv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaDoubleComplex_const_ptr dx,           magma_int_t incx,
    magmaDoubleComplex beta,             
    magmaDoubleComplex_ptr    dy,              magma_int_t incy,
    magmaDoubleComplex       *hwork,           magma_int_t lhwork,
    magmaDoubleComplex_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

magma_int_t
magmablas_zhemv_mgpu_sync(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda, magma_int_t offset,
    magmaDoubleComplex_const_ptr dx,           magma_int_t incx,
    magmaDoubleComplex beta,             
    magmaDoubleComplex_ptr    dy,              magma_int_t incy,
    magmaDoubleComplex       *hwork,           magma_int_t lhwork,
    magmaDoubleComplex_ptr    dwork[],         magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] );

// Ichi's version, in src/zhetrd_mgpu.cpp
void
magma_zher2k_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

void
magmablas_zher2k_mgpu_spec(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb, magma_queue_t queues[][20], magma_int_t nqueue );

void
magmablas_zher2k_mgpu2(
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[], magma_int_t ldda, magma_int_t a_offset,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue );

// in src/zpotrf_mgpu_right.cpp
void
magma_zherk_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );

// in src/zpotrf_mgpu_right.cpp
void
magma_zherk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10] );


  /*
   * LAPACK auxiliary functions (alphabetical order)
   */
void
magmablas_zgeadd_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_zgeadd2_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_zlacpy_q(
    magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_zlacpy_conj_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dA1, magma_int_t lda1,
    magmaDoubleComplex_ptr dA2, magma_int_t lda2,
    magma_queue_t queue );

void
magmablas_zlacpy_sym_in_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_zlacpy_sym_out_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_int_t *rows, magma_int_t *perm,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

double
magmablas_zlange_q(
    magma_norm_t norm,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

double
magmablas_zlanhe_q(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

double
magmablas_zlansy_q(
    magma_norm_t norm, magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_queue_t queue );

void
magmablas_zlarfg_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dalpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dtau,
    magma_queue_t queue );

void
magmablas_zlascl_q(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_zlascl_2x2_q(
    magma_type_t type, magma_int_t m,
    magmaDoubleComplex_const_ptr dW, magma_int_t lddw,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_zlascl2_q(
    magma_type_t type,
    magma_int_t m, magma_int_t n,
    magmaDouble_const_ptr dD,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_zlascl_diag_q(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dD, magma_int_t lddd,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info );

void
magmablas_zlaset_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_zlaset_band_q(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex offdiag, magmaDoubleComplex diag,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_zlaswp_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_zlaswp2_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dAT, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    magmaInt_const_ptr d_ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_zlaswp_sym_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_zlaswpx_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldx, magma_int_t ldy,
    magma_int_t k1, magma_int_t k2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_queue_t queue );

void
magmablas_zsymmetrize_q(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue );

void
magmablas_zsymmetrize_tiles_q(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue );

void
magmablas_ztrtri_diag_q(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr d_dinvA,
    magma_queue_t queue );

  /*
   * to cleanup (alphabetical order)
   */
magma_int_t
magma_zlaqps2_gpu_q(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr dvn1, magmaDouble_ptr dvn2,
    magmaDoubleComplex_ptr dauxv,
    magmaDoubleComplex_ptr dF,  magma_int_t lddf,
    magma_queue_t queue );

magma_int_t
magma_zlarfb_gpu_q(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT, magma_int_t lddt,
    magmaDoubleComplex_ptr dC,       magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,    magma_int_t ldwork,
    magma_queue_t queue );

magma_int_t
magma_zlarfb_gpu_gemm_q(
    magma_side_t side, magma_trans_t trans, magma_direct_t direct, magma_storev_t storev,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex_const_ptr dV, magma_int_t lddv,
    magmaDoubleComplex_const_ptr dT, magma_int_t lddt,
    magmaDoubleComplex_ptr dC,       magma_int_t lddc,
    magmaDoubleComplex_ptr dwork,    magma_int_t ldwork,
    magmaDoubleComplex_ptr dworkvt,  magma_int_t ldworkvt,
    magma_queue_t queue );

void
magma_zlarfbx_gpu_q(
    magma_int_t m, magma_int_t k,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr dT, magma_int_t ldt,
    magmaDoubleComplex_ptr c,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue );

void
magma_zlarfg_gpu_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dAkk,
    magma_queue_t queue );

void
magma_zlarfgtx_gpu_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue );

void
magma_zlarfgx_gpu_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magma_queue_t queue );

void
magma_zlarfx_gpu_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr tau,
    magmaDoubleComplex_ptr C, magma_int_t ldc,
    magmaDouble_ptr        xnorm,
    magmaDoubleComplex_ptr dT, magma_int_t iter,
    magmaDoubleComplex_ptr work,
    magma_queue_t queue );

  /*
   * Level 1 BLAS (alphabetical order)
   */
void
magmablas_zaxpycp_q(
    magma_int_t m,
    magmaDoubleComplex_ptr dr,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_const_ptr db,
    magma_queue_t queue );

void
magmablas_zswap_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

void
magmablas_zswapblk_q(
    magma_order_t order,
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr dB, magma_int_t lddb,
    magma_int_t i1, magma_int_t i2,
    const magma_int_t *ipiv, magma_int_t inci,
    magma_int_t offset,
    magma_queue_t queue );

void
magmablas_zswapdblk_q(
    magma_int_t n, magma_int_t nb,
    magmaDoubleComplex_ptr dA, magma_int_t ldda, magma_int_t inca,
    magmaDoubleComplex_ptr dB, magma_int_t lddb, magma_int_t incb,
    magma_queue_t queue );

void
magmablas_dznrm2_adjust_q(
    magma_int_t k,
    magmaDouble_ptr dxnorm,
    magmaDoubleComplex_ptr dc,
    magma_queue_t queue );

void
magmablas_dnrm2_check_q(
    magma_int_t m, magma_int_t n,
    magmaDouble_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_dznrm2_check_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dlsticc,
    magma_queue_t queue );

void
magmablas_dznrm2_cols_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dxnorm,
    magma_queue_t queue );

void
magmablas_dznrm2_row_check_adjust_q(
    magma_int_t k, double tol,
    magmaDouble_ptr dxnorm,
    magmaDouble_ptr dxnorm2,
    magmaDoubleComplex_ptr dC, magma_int_t lddc,
    magmaDouble_ptr dlsticc,
    magma_queue_t queue );

  /*
   * Level 2 BLAS (alphabetical order)
   */
// trsv were always queue versions
void
magmablas_ztrsv(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       db, magma_int_t incb,
    magma_queue_t queue );

// todo: move flag before queue?
void
magmablas_ztrsv_outofplace(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr db,       magma_int_t incb,
    magmaDoubleComplex_ptr dx,
    magma_queue_t queue,
    magma_int_t flag );

void
magmablas_ztrsv_work_batched(
    magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_ptr *dA_array, magma_int_t lda,
    magmaDoubleComplex_ptr *db_array, magma_int_t incb,
    magmaDoubleComplex_ptr *dx_array,
    magma_int_t batchCount,
    magma_queue_t queue );

void
magmablas_zgemv_q(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy, 
    magma_queue_t queue );

void
magmablas_zgemv_conj_q(
    magma_int_t m, magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_zhemv_q(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

magma_int_t
magmablas_zsymv_q(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

// hemv/symv_work were always queue versions
magma_int_t
magmablas_zhemv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

magma_int_t
magmablas_zsymv_work(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dwork, magma_int_t lwork,
    magma_queue_t queue );

  /*
   * Level 3 BLAS (alphabetical order)
   */
void
magmablas_zgemm_q(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zgemm_reduce_q(
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zhemm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zsymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zsyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zher2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zsyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_zherk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double  alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double  beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magmablas_ztrsm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magmablas_ztrsm_outofplace_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue );

void
magmablas_ztrsm_work_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magmaDoubleComplex_ptr       dX, magma_int_t lddx,
    magma_int_t flag,
    magmaDoubleComplex_ptr d_dinvA, magma_int_t dinvA_length,
    magma_queue_t queue );


  /*
   * Wrappers for platform independence.
   * These wrap CUBLAS or AMD OpenCL BLAS functions.
   */

// ========================================
// copying vectors
// set  copies host   to device
// get  copies device to host
// copy copies device to device
// (with CUDA unified addressing, copy can be between same or different devices)
// Add the function, file, and line for error-reporting purposes.

#define magma_zsetvector_q(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_zsetvector_q_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetvector_q(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_zgetvector_q_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zcopyvector_q(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_zcopyvector_q_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zsetvector_async(           n, hx_src, incx, dy_dst, incy, queue ) \
        magma_zsetvector_async_internal(  n, hx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetvector_async(           n, dx_src, incx, hy_dst, incy, queue ) \
        magma_zgetvector_async_internal(  n, dx_src, incx, hy_dst, incy, queue, __func__, __FILE__, __LINE__ )

#define magma_zcopyvector_async(          n, dx_src, incx, dy_dst, incy, queue ) \
        magma_zcopyvector_async_internal( n, dx_src, incx, dy_dst, incy, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_zsetvector_q_internal(
    magma_int_t n,
    magmaDoubleComplex const    *hx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_q_internal( n, sizeof(magmaDoubleComplex), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_zgetvector_q_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_q_internal( n, sizeof(magmaDoubleComplex), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void
magma_zcopyvector_q_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_q_internal( n, sizeof(magmaDoubleComplex), dx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_zsetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex const    *hx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setvector_async_internal( n, sizeof(magmaDoubleComplex), hx_src, incx, dy_dst, incy, queue, func, file, line ); }

static inline void
magma_zgetvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex          *hy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getvector_async_internal( n, sizeof(magmaDoubleComplex), dx_src, incx, hy_dst, incy, queue, func, file, line ); }

static inline void
magma_zcopyvector_async_internal(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx_src, magma_int_t incx,
    magmaDoubleComplex_ptr       dy_dst, magma_int_t incy,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copyvector_async_internal( n, sizeof(magmaDoubleComplex), dx_src, incx, dy_dst, incy, queue, func, file, line ); }


// ========================================
// copying sub-matrices (contiguous columns)

#define magma_zsetmatrix_q(           m, n, hA_src, lda,  dB_dst, lddb, queue ) \
        magma_zsetmatrix_q_internal(  m, n, hA_src, lda,  dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix_q(           m, n, dA_src, ldda, hB_dst, ldb,  queue ) \
        magma_zgetmatrix_q_internal(  m, n, dA_src, ldda, hB_dst, ldb,  queue, __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix_q(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_zcopymatrix_q_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_zsetmatrix_async(           m, n, hA_src, lda, dB_dst, lddb, queue ) \
        magma_zsetmatrix_async_internal(  m, n, hA_src, lda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

#define magma_zgetmatrix_async(           m, n, dA_src, ldda, hB_dst, ldb, queue ) \
        magma_zgetmatrix_async_internal(  m, n, dA_src, ldda, hB_dst, ldb, queue, __func__, __FILE__, __LINE__ )

#define magma_zcopymatrix_async(          m, n, dA_src, ldda, dB_dst, lddb, queue ) \
        magma_zcopymatrix_async_internal( m, n, dA_src, ldda, dB_dst, lddb, queue, __func__, __FILE__, __LINE__ )

static inline void
magma_zsetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const    *hA_src, magma_int_t lda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_q_internal( m, n, sizeof(magmaDoubleComplex), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_zgetmatrix_q_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_q_internal( m, n, sizeof(magmaDoubleComplex), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void
magma_zcopymatrix_q_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_q_internal( m, n, sizeof(magmaDoubleComplex), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_zsetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex const    *hA_src, magma_int_t lda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_setmatrix_async_internal( m, n, sizeof(magmaDoubleComplex), hA_src, lda, dB_dst, lddb, queue, func, file, line ); }

static inline void
magma_zgetmatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex          *hB_dst, magma_int_t ldb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_getmatrix_async_internal( m, n, sizeof(magmaDoubleComplex), dA_src, ldda, hB_dst, ldb, queue, func, file, line ); }

static inline void
magma_zcopymatrix_async_internal(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA_src, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB_dst, magma_int_t lddb,
    magma_queue_t queue,
    const char* func, const char* file, int line )
{ magma_copymatrix_async_internal( m, n, sizeof(magmaDoubleComplex), dA_src, ldda, dB_dst, lddb, queue, func, file, line ); }


// ========================================
// Level 1 BLAS (alphabetical order)

// in cublas_v2, result returned through output argument
magma_int_t
magma_izamax_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
magma_int_t
magma_izamin_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
double
magma_dzasum_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_zaxpy_q(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_zcopy_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotc_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
magmaDoubleComplex
magma_zdotu_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// in cublas_v2, result returned through output argument
double
magma_dznrm2_q(
    magma_int_t n,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_zrot_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, magmaDoubleComplex ds,
    magma_queue_t queue );

void
magma_zdrot_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    double dc, double ds,
    magma_queue_t queue );

#ifdef REAL
void
magma_zrotm_q(
    magma_int_t n,
    magmaDouble_ptr dx, magma_int_t incx,
    magmaDouble_ptr dy, magma_int_t incy,
    magmaDouble_const_ptr param,
    magma_queue_t queue );

void
magma_zrotmg_q(
    magmaDouble_ptr d1, magmaDouble_ptr       d2,
    magmaDouble_ptr x1, magmaDouble_const_ptr y1,
    magmaDouble_ptr param,
    magma_queue_t queue );
#endif

void
magma_zscal_q(
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_zdscal_q(
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_zswap_q(
    magma_int_t n,
    magmaDoubleComplex_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue );

// ========================================
// Level 2 BLAS (alphabetical order)

void
magma_zgemv_q(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_zgerc_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_zgeru_q(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_zhemv_q(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dy, magma_int_t incy,
    magma_queue_t queue );

void
magma_zher_q(
    magma_uplo_t uplo,
    magma_int_t n,
    double alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_zher2_q(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dx, magma_int_t incx,
    magmaDoubleComplex_const_ptr dy, magma_int_t incy,
    magmaDoubleComplex_ptr       dA, magma_int_t ldda,
    magma_queue_t queue );

void
magma_ztrmv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

void
magma_ztrsv_q(
    magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dx, magma_int_t incx,
    magma_queue_t queue );

// ========================================
// Level 3 BLAS (alphabetical order)

void
magma_zgemm_q(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zsymm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zhemm_q(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zsyr2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zher2k_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_const_ptr dB, magma_int_t lddb,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zsyrk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_zherk_q(
    magma_uplo_t uplo, magma_trans_t trans,
    magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    double beta,
    magmaDoubleComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue );

void
magma_ztrmm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );

void
magma_ztrsm_q(
    magma_side_t side, magma_uplo_t uplo, magma_trans_t trans, magma_diag_t diag,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDoubleComplex_ptr       dB, magma_int_t lddb,
    magma_queue_t queue );


#ifdef __cplusplus
}
#endif

#undef COMPLEX

#endif  /* MAGMABLAS_Z_H */
