/*
   -- MAGMA (version 1.1) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   November 2011

   @precisions normal z -> s d c

*/

#include "common_magma.h"
#include <cblas.h>

#define PRECISION_z


extern "C" magma_int_t 
magma_zlaqps(magma_int_t *m, magma_int_t *n, magma_int_t *offset, 
             magma_int_t *nb, magma_int_t *kb, 
             cuDoubleComplex *a, magma_int_t *lda,
             cuDoubleComplex *da, magma_int_t *ldda,
             magma_int_t *jpvt, cuDoubleComplex *tau, double *vn1, double *vn2, 
             cuDoubleComplex *auxv, 
             cuDoubleComplex *f, magma_int_t *ldf,
             cuDoubleComplex *df, magma_int_t *lddf)
{
/*
    -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2011

    Purpose   
    =======   
    ZLAQPS computes a step of QR factorization with column pivoting   
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize   
    NB columns from A starting from the row OFFSET+1, and updates all   
    of the matrix with Blas-3 xGEMM.   

    In some cases, due to catastrophic cancellations, it cannot   
    factorize NB columns.  Hence, the actual number of factorized   
    columns is returned in KB.   

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.   

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows of the matrix A. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A. N >= 0   

    OFFSET  (input) INTEGER   
            The number of rows of A that have been factorized in   
            previous steps.   

    NB      (input) INTEGER   
            The number of columns to factorize.   

    KB      (output) INTEGER   
            The number of columns actually factorized.   

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, block A(OFFSET+1:M,1:KB) is the triangular   
            factor obtained and block A(1:OFFSET,1:N) has been   
            accordingly pivoted, but no factorized.   
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has   
            been updated.   

    LDA     (input) INTEGER   
            The leading dimension of the array A. LDA >= max(1,M).   

    JPVT    (input/output) INTEGER array, dimension (N)   
            JPVT(I) = K <==> Column K of the full matrix A has been   
            permuted into position I in AP.   

    TAU     (output) COMPLEX*16 array, dimension (KB)   
            The scalar factors of the elementary reflectors.   

    VN1     (input/output) DOUBLE PRECISION array, dimension (N)   
            The vector with the partial column norms.   

    VN2     (input/output) DOUBLE PRECISION array, dimension (N)   
            The vector with the exact column norms.   

    AUXV    (input/output) COMPLEX*16 array, dimension (NB)   
            Auxiliar vector.   

    F       (input/output) COMPLEX*16 array, dimension (LDF,NB)   
            Matrix F' = L*Y'*A.   

    LDF     (input) INTEGER   
            The leading dimension of the array F. LDF >= max(1,N).   

    Further Details   
    ===============   

    Based on contributions by   
      G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain   
      X. Sun, Computer Science Dept., Duke University, USA   
    =====================================================================    */
    
#define  A(i, j) (a    +(j)*(*lda)  + (i))
#define dA(i, j) (da   +(j)*(*ldda) + (i))

    static cuDoubleComplex zero = MAGMA_Z_MAKE(0.,0.);
    static cuDoubleComplex one  = MAGMA_Z_MAKE(1.,0.);
    static cuDoubleComplex mone = MAGMA_Z_MAKE(-1.,0.);
    static magma_int_t c__1 = 1;
    
    magma_int_t a_dim1, a_offset, f_dim1, f_offset, i__1, i__2;
    double d__1;
    cuDoubleComplex z__1;
    
    static magma_int_t j, k, rk;
    static cuDoubleComplex akk;
    static magma_int_t pvt;
    static double temp, temp2, tol3z;
    static magma_int_t itemp;

    static magma_int_t lsticc;
    static magma_int_t lastrk;    
    
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --jpvt;
    --tau;
    --vn1;
    --vn2;
    --auxv;
    f_dim1 = *ldf;
    f_offset = 1 + f_dim1;
    f -= f_offset;

    i__1 = *m, i__2 = *n + *offset;
    lastrk = min(i__1,i__2);
    lsticc = 0;
    k = 0;
    tol3z = magma_dsqrt(dlamch_("Epsilon"));

    da -= 1 + *ldda;
    df -= 1 + *lddf;

    static cudaStream_t stream;
    magma_queue_create( &stream );

    // TTT =========================================================
    /*
    rk = *offset + *kb;
    i__1 = *m - rk;
    i__2 = *n - *kb;
    if (*kb < min(*n, *m - *offset))
        fprintf(stderr,"%e ", 
                cpu_gpu_zdiff(i__1, i__2, &a[rk + 1 + (*kb + 1) * a_dim1], *lda,
                              &da[rk + 1 + (*kb + 1) * a_dim1], *ldda));
    */
    // TTT =========================================================

 L10:
    if (k < *nb && lsticc == 0) {
        ++k;
        rk = *offset + k;
        
        /* Determine ith pivot column and swap if necessary */
        i__1 = *n - k + 1;

        /* Comment:
           Fortran BLAS does not have to add 1
           C       BLAS must add one to cblas_idamax */
        //pvt = k - 1 + idamax_(&i__1, &vn1[k], &c__1);
        pvt = k - 1 + cblas_idamax(i__1, &vn1[k], c__1) +1;

        if (pvt != k) {
            
            //blasf77_zswap(m, &a[pvt * a_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &c__1);
            if (pvt <= *nb){
                // TTT - no need of transfer if pivot is within the panel 
                blasf77_zswap(m, &a[pvt * a_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &c__1);
            }
            else {
                // TTT - first start copy from GPU
                magma_zgetmatrix( *m - *offset - *nb, 1,
                                  dA(*offset + *nb+1,pvt), *ldda,
                                  A (*offset + *nb+1,pvt), *lda );
                
                // TTT - swap as usual on CPU
                blasf77_zswap(m, &a[pvt * a_dim1 + 1], &c__1, &a[k * a_dim1 + 1], &c__1);
                // TTT - restore the GPU
                magma_zsetmatrix_async( *m - *offset - *nb, 1,
                                        A (*offset + *nb+1,pvt), *lda,
                                        dA(*offset + *nb+1,pvt), *ldda, stream);
            }

            i__1 = k - 1;

            /* TTT - this forces us to send F at the end to GPU       */
            blasf77_zswap(&i__1, &f[pvt + f_dim1], ldf, &f[k + f_dim1], ldf);
            itemp = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k] = itemp;
            vn1[pvt] = vn1[k];
            vn2[pvt] = vn2[k];
        }

        /* Apply previous Householder reflectors to column K:   
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'. 
           TTT - can multiply with 0 ; vector has to arrive and subtract */
        if (k > 1) {
            #if (defined(PRECISION_c) || defined(PRECISION_z))
            for (j = 1; j < k; ++j){
                f[ k + j * f_dim1 ] = MAGMA_Z_CNJG(f[k + j * f_dim1]);
            }
            #endif

            i__1 = *m - rk + 1;
            i__2 = k - 1;
            blasf77_zgemv(MagmaNoTransStr, &i__1, &i__2, &mone, &a[rk + a_dim1], lda, 
                          &f[k + f_dim1], ldf, &one, &a[rk + k * a_dim1], &c__1);

            #if (defined(PRECISION_c) || defined(PRECISION_z))
            for (j = 1; j < k; ++j) {
                f[k + j * f_dim1] =MAGMA_Z_CNJG( f[k + j * f_dim1] );
            }
            #endif
        }
        
        /*  Generate elementary reflector H(k). */
        if (rk < *m) {
            i__1 = *m - rk + 1;
            lapackf77_zlarfg(&i__1, &a[rk + k * a_dim1], &a[rk + 1 + k * a_dim1], &
                             c__1, &tau[k]);
        } else {
            lapackf77_zlarfg(&c__1, &a[rk + k * a_dim1], &a[rk + k * a_dim1], &c__1, &
                             tau[k]);
        }
        
        akk = a[rk + k * a_dim1];
        a[rk + k * a_dim1] = one;

       /* Compute Kth column of F:   
          Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K). 
          TTT - on the GPU */
       if (k < *n) {
           i__1 = *m - rk + 1;
           i__2 = *n - k;
           /*
           blasf77_zgemv(MagmaConjTransStr, &i__1, &i__2, &tau[k], 
                         A(rk, k + 1), lda, A(rk, k), 
                         &c__1, &zero, &f[k + 1 + k * f_dim1], &c__1);
           */
           
           // TTT - Send the vector to the GPU
           magma_zsetmatrix( i__1, 1, A(rk, k), *lda, dA(rk,k), *ldda );

           // TTT - multiply on GPU
           int i__3 = *nb-k;
           int i__4 = i__2 -i__3, i__5 = *nb-k+1;

           magma_zgemv(MagmaConjTrans, i__1 - i__5, i__2 - i__3, tau[k],
                       dA(rk+i__5, k+1+i__3), *ldda, dA(rk+i__5, k),
                       c__1, zero, &df[k + 1 +i__3 + k * f_dim1], c__1);

           magma_zgetmatrix_async(i__2-i__3, 1,
                                  &df[k + 1 +i__3+k * f_dim1], i__2,
                                  & f[k + 1 +i__3+k * f_dim1], i__2, stream);

           blasf77_zgemv(MagmaConjTransStr, &i__1, &i__3, &tau[k],
                         A(rk, k + 1), lda, A(rk, k),
                         &c__1, &zero, &f[k + 1 + k * f_dim1], &c__1);

           magma_queue_sync( stream );
           blasf77_zgemv(MagmaConjTransStr, &i__5, &i__4, &tau[k],
                         A(rk, k+1 +i__3), lda, A(rk, k),
                         &c__1, &one, &f[k + 1 + i__3 + k * f_dim1], &c__1);

           //fprintf(stderr,"||e|| = %e\n", cpu_gpu_zdiff(i__1 + (k-*nb)-1, i__2 +(k-*nb), 
           //                                           A(rk-(k-*nb)+1, k+1-(k-*nb)), *lda,
           //                                           dA(rk-(k-*nb)+1, k+1-(k-*nb)), *ldda));

           
       }
       
       /* Padding F(1:K,K) with zeros. */
       for (j = 1; j <= k; ++j)
           f[j + k * f_dim1] = zero;
       
       /* Incremental updating of F:   
          F(1:N,K) := F(1:N,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K). 
          TTT - result has to come back while computing the vector - to just subtract */
       if (k > 1) {
           i__1 = *m - rk + 1;
           i__2 = k - 1;
           double temporary1 = MAGMA_Z_REAL(tau[k]);
           double temporary2 = MAGMA_Z_IMAG(tau[k]);
           z__1 = MAGMA_Z_MAKE(-temporary1, -temporary2);
           blasf77_zgemv(MagmaConjTransStr, &i__1, &i__2, &z__1, &a[rk + a_dim1],
                         lda, &a[rk + k * a_dim1], &c__1, &zero, &auxv[1], &c__1);
           
           i__1 = k - 1;
           blasf77_zgemv(MagmaNoTransStr, n, &i__1, &one, &f[f_dim1 + 1], ldf, 
                         &auxv[1], &c__1, &one, &f[k * f_dim1 + 1], &c__1);
       }
       
       /* TTT -- On the last iteration we can start sending F back to the GPU */

       /* Update the current row of A:   
          A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'. 
          TTT - can also be split if needed to wait for the product on GPU */
       if (k < *n) {
           i__1 = *n - k;
           blasf77_zgemm(MagmaNoTransStr, MagmaConjTransStr, &c__1, &i__1, &k,
                         &mone, &a[rk + a_dim1], lda, &f[k + 1 + f_dim1], ldf,
                         &one, &a[rk + (k + 1) * a_dim1], lda);
       }
       
       /* Update partial column norms. */
       if (rk < lastrk) {
           for (j = k + 1; j <= *n; ++j) {
               if (vn1[j] != 0.) {                   
                   /* NOTE: The following 4 lines follow from the analysis in   
                      Lapack Working Note 176. */
                   temp = MAGMA_Z_ABS(a[rk + j * a_dim1]) / vn1[j];

                   temp = max(0., ((temp + 1.) * (1. - temp)));

                   d__1 = vn1[j] / vn2[j];
                   temp2 = temp * (d__1 * d__1);
                   if (temp2 <= tol3z) {
                       vn2[j] = (double) lsticc;
                       lsticc = j;
                   } else {
                       vn1[j] *= magma_dsqrt(temp);
                   }
               }
           }
       }
       
       a[rk + k * a_dim1] = akk;
       
       /* End of while loop. */
       goto L10;
    }
    *kb = k;
    rk = *offset + *kb;

    /* Apply the block reflector to the rest of the matrix:   
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(*n, *m - *offset)) {
       i__1 = *m - rk;
       i__2 = *n - *kb;
       //fprintf(stderr,"difference %3d %3d %e", i__1, i__2,
       //      cpu_gpu_zdiff(i__1, i__2, &a[rk + 1 + (*kb + 1) * a_dim1], *lda,
       //                    &da[rk + 1 + (*kb + 1) * a_dim1], *ldda));
       
       //blasf77_zgemm(MagmaNoTransStr, MagmaConjTransStr, &i__1, &i__2, kb, &mone, 
       //              &a[rk + 1 + a_dim1], lda, &f[*kb + 1 + f_dim1], ldf, &one,
       //              &a[rk + 1 + (*kb + 1) * a_dim1], lda);
       
       // TTT - send f to the GPU
       magma_zsetmatrix( i__2, *kb,
                         &f [*kb + 1 + f_dim1], *ldf,
                         &df[*kb + 1 + f_dim1], i__2);
       // TTT - send V to the GPU (is it there or was it modified, i.e., is this needed?
       magma_zsetmatrix( i__1, *kb,
                         &a [rk + 1 + a_dim1], *lda,
                         &da[rk + 1 + a_dim1], *ldda);

       magma_zgemm(MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb, mone,
                   &da[rk + 1 + a_dim1], *ldda, &df[*kb + 1 + f_dim1], i__2, one,
                   &da[rk + 1 + (*kb + 1) * a_dim1], *ldda);
       //fprintf(stderr," %e\n", 
       //        cpu_gpu_zdiff(i__1, i__2, &a[rk + 1 + (*kb + 1) * a_dim1], *lda,
       //                      &da[rk + 1 + (*kb + 1) * a_dim1], *ldda));
    }
    
    /* Recomputation of difficult columns. */
 L60:
    if (lsticc > 0) {
       itemp = (magma_int_t)(vn2[lsticc] >= 0. ? floor(vn2[lsticc] + .5) : -floor(.5 - vn2[lsticc]));  
       i__1 = *m - rk;
       vn1[lsticc] = cblas_dznrm2(i__1, &a[rk + 1 + lsticc * a_dim1], c__1);

       // TTT (where is the data, CPU or GPU ?)
       vn1[lsticc] = cublasDznrm2(i__1, &da[rk + 1 + lsticc * a_dim1], c__1);
       printf("lsticc = %d\n", lsticc);
       /* NOTE: The computation of VN1( LSTICC ) relies on the fact that   
          SNRM2 does not fail on vectors with norm below the value of SQRT(DLAMCH('S')) */       
       vn2[lsticc] = vn1[lsticc];
       lsticc = itemp;
       goto L60;
    }
    
    magma_queue_destroy( stream );

    return MAGMA_SUCCESS;
} /* magma_zlaqps */
