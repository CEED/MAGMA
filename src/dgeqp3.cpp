/*
  -- MAGMA (version 1.1) --
     Univ. of Tennessee, Knoxville
     Univ. of California, Berkeley
     Univ. of Colorado, Denver
     November 2011

     @precisions normal d -> s

*/

#include "common_magma.h"
#include <cblas.h>

#define lapackf77_dlaqps   FORTRAN_NAME( dlaqps, DLAQPS)
#define lapackf77_dlaqp2   FORTRAN_NAME( zlaqp2, ZLAQP2)

extern "C" int 
lapackf77_dlaqps(magma_int_t *m, magma_int_t *n, magma_int_t *offset, magma_int_t
                 *nb, magma_int_t *kb, double *a, magma_int_t *lda, magma_int_t *jpvt,
                 double *tau, double *vn1, double *vn2, double *
                 auxv, double *f, magma_int_t *ldf);

extern "C" void 
lapackf77_dlaqp2(magma_int_t *m, magma_int_t *n, magma_int_t *offset, 
                 double *a, magma_int_t *lda, magma_int_t *jpvt, double *tau,
                 double *vn1, double *vn2, double *work);

extern "C" int 
magma_dlaqps(magma_int_t *m, magma_int_t *n, magma_int_t *offset, magma_int_t
             *nb, magma_int_t *kb, double *a, magma_int_t *lda, magma_int_t *jpvt,
             double *tau, double *vn1, double *vn2, double *
             auxv, double *f, magma_int_t *ldf);

extern "C" magma_int_t 
magma_dgeqp3(magma_int_t *m, magma_int_t *n, double *a, 
             magma_int_t *lda, magma_int_t *jpvt, double *tau, double *work, 
             magma_int_t *lwork, magma_int_t *info)
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011
       
    Purpose   
    =======   
    DGEQP3 computes a QR factorization with column pivoting of a   
    matrix A:  A*P = Q*R  using Level 3 BLAS.   

    Arguments   
    =========   
    M       (input) INTEGER
            The number of rows of the matrix A. M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) COMPLEX*16 array, dimension (LDA,N)   
            On entry, the M-by-N matrix A.   
            On exit, the upper triangle of the array contains the   
            min(M,N)-by-N upper trapezoidal matrix R; the elements below   
            the diagonal, together with the array TAU, represent the   
            orthogonal matrix Q as a product of min(M,N) elementary   
            reflectors.   

    LDA     (input) INTEGER   
            The leading dimension of the array A. LDA >= max(1,M).   

    JPVT    (input/output) INTEGER array, dimension (N)   
            On entry, if JPVT(J).ne.0, the J-th column of A is permuted   
            to the front of A*P (a leading column); if JPVT(J)=0,   
            the J-th column of A is a free column.   
            On exit, if JPVT(J)=K, then the J-th column of A*P was the   
            the K-th column of A.   

    TAU     (output) COMPLEX*16 array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors.   

    WORK    (workspace/output) COMPLEX*16 array, dimension (MAX(1,LWORK))   
            On exit, if INFO=0, WORK(1) returns the optimal LWORK.   

    LWORK   (input) INTEGER   
            The dimension of the array WORK. LWORK >= 3*N+1.   
            For optimal performance LWORK >= 2*N + ( N+1 )*NB, where NB   
            is the optimal blocksize.   

            If LWORK = -1, then a workspace query is assumed; the routine   
            only calculates the optimal size of the WORK array, returns   
            this value as the first entry of the WORK array, and no error   
            message related to LWORK is issued by XERBLA.   

    INFO    (output) INTEGER   
            = 0: successful exit.   
            < 0: if INFO = -i, the i-th argument had an illegal value.   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

      Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

      H(i) = I - tau * v * v'   

    where tau is a real/real scalar, and v is a real/real vector   
    with v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in   
    A(i+1:m,i), and tau in TAU(i).   

    Based on contributions by   
      G. Quintana-Orti, Depto. de Informatica, Universidad Jaime I, Spain   
      X. Sun, Computer Science Dept., Duke University, USA   
    =====================================================================   */

    static magma_int_t c__1 = 1;
    static magma_int_t c_n1 = -1;
    static magma_int_t c__3 = 3;
    
    magma_int_t a_dim1, a_offset, i__1, i__2, i__3;
    static magma_int_t j, jb, na, nb, sm, sn, nx, fjb, iws, nfxd, nbmin, minmn, minws;

    static magma_int_t topbmn, sminmn;
    static magma_int_t lwkopt;
    static magma_int_t lquery;
    
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --jpvt;
    --tau;
    --work;
    
    *info = 0;
    lquery = *lwork == -1;
    if (*m < 0) {
       *info = -1;
    } else if (*n < 0) {
       *info = -2;
    } else if (*lda < max(1,*m)) {
       *info = -4;
    }
    
    if (*info == 0) {
        minmn = min(*m,*n);
        if (minmn == 0) {
            iws = 1;
            lwkopt = 1;
        } else {
            iws = *n + 1;
            nb = magma_get_dgeqrf_nb(min(*m, *n));
            lwkopt = 2*(*n) + (*n + 1) * nb;
        }
        MAGMA_D_SET2REAL(work[1],(double)lwkopt);

        if (*lwork < iws && ! lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        i__1 = -(*info);
        magma_xerbla( __func__, -(*info) );
        return *info;
    } else if (lquery) {
        return *info;
    }

    /* Quick return if possible. */
    if (minmn == 0) {
        return *info;
    }

    /* Move initial columns up front. */
    nfxd = 1;
    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
        if (jpvt[j] != 0) {
           if (j != nfxd) {
               blasf77_dswap(m, &a[j * a_dim1 + 1], &c__1, &a[nfxd * a_dim1 + 1], &c__1);
               jpvt[j] = jpvt[nfxd];
               jpvt[nfxd] = j;
           } else {
               jpvt[j] = j;
           }
           ++nfxd;
        } else {
            jpvt[j] = j;
        }
    }
    --nfxd;

    /*     Factorize fixed columns   
           =======================   
           Compute the QR factorization of fixed columns and update   
           remaining columns. */
    if (nfxd > 0) {
        na = min(*m,nfxd);
        lapackf77_dgeqrf(m, &na, &a[a_offset], lda, &tau[1], &work[1], lwork, info);

        /* Computing MAX */
        i__1 = iws, i__2 = (magma_int_t) (work[1]);
        iws = max(i__1,i__2);
        if (na < *n) {
            i__1 = *n - na;
            lapackf77_dormqr(MagmaLeftStr, MagmaTransStr, m, &i__1, &na, &a[a_offset],
                             lda, &tau[1], &a[(na + 1) * a_dim1 + 1], lda, &work[1], 
                             lwork, info);
            /* Computing MAX */
            i__1 = iws, i__2 = (magma_int_t  ) (work[1]);
            iws = max(i__1,i__2);
        }
    }
    
    /*  Factorize free columns */
    if (nfxd < minmn) {
       sm = *m - nfxd;
       sn = *n - nfxd;
       sminmn = minmn - nfxd;

       /* Determine the block size. */
       nbmin = 2;
       nx = 0;

       if (nb > 1 && nb < sminmn) {

           /* Determine when to cross over from blocked to unblocked code */
           nx = nb;

           if (nx < sminmn) {               
               /* Determine if workspace is large enough for blocked code. */
               minws = 2*sn + (sn + 1) * nb;
               iws = max(iws,minws);
               if (*lwork < minws) {
       
                   /* Not enough workspace to use optimal NB: Reduce NB and   
                      determine the minimum value of NB. */
                   nb = (*lwork - 2*sn)/ (sn + 1);
                   nbmin = max(i__1,i__2);
               }
           }
       }

       /*       Initialize partial column norms. The first N elements of work   
                store the exact column norms. */
       i__1 = *n;
       for (j = nfxd + 1; j <= i__1; ++j) {
           work[j] = cblas_dnrm2(sm, &a[nfxd + 1 + j * a_dim1], c__1);
           work[*n + j] = work[j];
       }
       
       if (nb >= nbmin && nb < sminmn && nx < sminmn) {
           
           /* Use blocked code initially. */   
           j = nfxd + 1;
           
           /* Compute factorization: while loop. */
           topbmn = minmn - nx;
       L30:
           if (j <= topbmn) {
               /* Computing MIN */
               i__1 = nb, i__2 = topbmn - j + 1;
               jb = min(i__1,i__2);
               
               /*            Factorize JB columns among columns J:N. */
               
               i__1 = *n - j + 1;
               i__2 = j - 1;
               i__3 = *n - j + 1;
               //lapackf77_dlaqps(m, &i__1, &i__2, &jb, &fjb, &a[j * a_dim1 + 1], lda, 
               magma_dlaqps(m, &i__1, &i__2, &jb, &fjb, &a[j * a_dim1 + 1], lda, 
                            &jpvt[j], &tau[j], &work[j], &work[*n + j], &work[2*(*n)+1],
                            &work[2*(*n) + jb + 1], &i__3);
               
               j += fjb;
               goto L30;
           }
       } else {
           j = nfxd + 1;
       }
        
       /* Use unblocked code to factor the last or only block. */
       if (j <= minmn) {
           i__1 = *n - j + 1;
           i__2 = j - 1;
           lapackf77_dlaqp2(m, &i__1, &i__2, &a[j * a_dim1 + 1], lda, &jpvt[j], 
                            &tau[j], &work[j], &work[*n + j], &work[2*(*n) + 1]);
       }
    }

    work[1] = MAGMA_D_MAKE((double) iws, 0.);
    return *info;
} /* dgeqp3_ */
