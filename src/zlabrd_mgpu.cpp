/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Azzam Haidar
       @author Yulu Jia       
       @precisions normal z -> s d c

*/
#include "common_magma.h"

#include <assert.h>

#define PRECISION_z
//#define magma_zgemv magmablas_zgemv

/**
    Purpose
    -------
    ZLABRD reduces the first NB rows and columns of a complex general
    m by n matrix A to upper or lower bidiagonal form by an orthogonal
    transformation Q' * A * P, and returns the matrices X and Y which
    are needed to apply the transformation to the unreduced part of A.

    If m >= n, A is reduced to upper bidiagonal form; if m < n, to lower
    bidiagonal form.

    This is an auxiliary routine called by ZGEBRD.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows in the matrix A.

    @param[in]
    n       INTEGER
            The number of columns in the matrix A.

    @param[in]
    nb      INTEGER
            The number of leading rows and columns of A to be reduced.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the m by n general matrix to be reduced.
            On exit, the first NB rows and columns of the matrix are
            overwritten; the rest of the array is unchanged.
            If m >= n, elements on and below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors; and
              elements above the diagonal in the first NB rows, with the
              array TAUP, represent the orthogonal matrix P as a product
              of elementary reflectors.
    \n
            If m < n, elements below the diagonal in the first NB
              columns, with the array TAUQ, represent the orthogonal
              matrix Q as a product of elementary reflectors, and
              elements on and above the diagonal in the first NB rows,
              with the array TAUP, represent the orthogonal matrix P as
              a product of elementary reflectors.
            See Further Details.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            Copy of A on GPU.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[out]
    d       COMPLEX_16 array, dimension (NB)
            The diagonal elements of the first NB rows and columns of
            the reduced matrix.  D(i) = A(i,i).

    @param[out]
    e       COMPLEX_16 array, dimension (NB)
            The off-diagonal elements of the first NB rows and columns of
            the reduced matrix.

    @param[out]
    tauq    COMPLEX_16 array dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix Q. See Further Details.

    @param[out]
    taup    COMPLEX_16 array, dimension (NB)
            The scalar factors of the elementary reflectors which
            represent the orthogonal matrix P. See Further Details.

    @param[out]
    X       COMPLEX_16 array, dimension (LDX,NB)
            The m-by-nb matrix X required to update the unreduced part
            of A.

    @param[in]
    ldx     INTEGER
            The leading dimension of the array X. LDX >= M.

    @param[out]
    dX      COMPLEX_16 array, dimension (LDDX,NB)
            Copy of X on GPU.

    @param[in]
    lddx    INTEGER
            The leading dimension of the array dX. LDDX >= M.

    @param[out]
    Y       COMPLEX_16 array, dimension (LDY,NB)
            The n-by-nb matrix Y required to update the unreduced part
            of A.

    @param[in]
    ldy     INTEGER
            The leading dimension of the array Y. LDY >= N.

    @param[out]
    dY      COMPLEX_16 array, dimension (LDDY,NB)
            Copy of Y on GPU.

    @param[in]
    lddy    INTEGER
            The leading dimension of the array dY. LDDY >= N.

    Further Details
    ---------------
    The matrices Q and P are represented as products of elementary
    reflectors:

       Q = H(1) H(2) . . . H(nb)  and  P = G(1) G(2) . . . G(nb)

    Each H(i) and G(i) has the form:

       H(i) = I - tauq * v * v'  and G(i) = I - taup * u * u'

    where tauq and taup are complex scalars, and v and u are complex vectors.

    If m >= n, v(1:i-1) = 0, v(i) = 1, and v(i:m) is stored on exit in
    A(i:m,i); u(1:i) = 0, u(i+1) = 1, and u(i+1:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    If m < n, v(1:i) = 0, v(i+1) = 1, and v(i+1:m) is stored on exit in
    A(i+2:m,i); u(1:i-1) = 0, u(i) = 1, and u(i:n) is stored on exit in
    A(i,i+1:n); tauq is stored in TAUQ(i) and taup in TAUP(i).

    The elements of the vectors v and u together form the m-by-nb matrix
    V and the nb-by-n matrix U' which are needed, with X and Y, to apply
    the transformation to the unreduced part of the matrix, using a block
    update of the form:  A := A - V*Y' - X*U'.

    The contents of A on exit are illustrated by the following examples
    with nb = 2:

    @verbatim
    m = 6 and n = 5 (m > n):          m = 5 and n = 6 (m < n):

      (  1   1   u1  u1  u1 )           (  1   u1  u1  u1  u1  u1 )
      (  v1  1   1   u2  u2 )           (  1   1   u2  u2  u2  u2 )
      (  v1  v2  a   a   a  )           (  v1  1   a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )           (  v1  v2  a   a   a   a  )
      (  v1  v2  a   a   a  )
    @endverbatim

    where a denotes an element of the original matrix which is unchanged,
    vi denotes an element of the vector defining H(i), and ui an element
    of the vector defining G(i).

    @ingroup magma_zgesvd_aux
    ********************************************************************/
extern "C" magma_int_t
magma_zlabrd_mgpu( magma_int_t ngpu, magma_int_t gbi, 
                  magma_int_t m, magma_int_t n, magma_int_t nb,
                  magmaDoubleComplex *A,  magma_int_t lda,
                  magmaDoubleComplex **dA, magma_int_t ldda,
                  double *d, double *e, magmaDoubleComplex *tauq, magmaDoubleComplex *taup,
                  magmaDoubleComplex *X,  magma_int_t ldx,
                  magmaDoubleComplex *dX, magma_int_t lddx,
                  magmaDoubleComplex *Y,  magma_int_t ldy,
                  magmaDoubleComplex *dY, magma_int_t lddy)
{
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one = MAGMA_Z_ONE;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t c__1 = 1;
    
    magma_int_t a_dim1, a_offset, x_dim1, x_offset, y_dim1, y_offset, i__2, i__3;
    magma_int_t i__;
    magmaDoubleComplex alpha;

    a_dim1 = lda;
    a_offset = 1 + a_dim1;
    A -= a_offset;
    --d;
    --e;
    --tauq;
    --taup;

    x_dim1 = ldx;
    x_offset = 1 + x_dim1;
    X -= x_offset;
    dX -= 1 + lddx;

    y_dim1 = ldy;
    y_offset = 1 + y_dim1;
    Y -= y_offset;
    dY -= 1 + lddy;

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return 0;
    }

    
    --gbi; //make global_i as C indexing
    magmaDoubleComplex *dwork[ngpu], *dwork2[ngpu];
    magma_int_t k, ib, dev, blkid, pos, devid, prev_devid, myn, loc_colindex;
    magma_int_t nlocal[ngpu], colindex_ptr[ngpu], firstelement_ptr[ngpu];
    memset(nlocal,0,ngpu*sizeof(magma_int_t));
    magma_int_t first_devid = (gbi/nb)%ngpu;  
    magma_int_t nblk       = n/nb;
    magma_int_t        nloc       = (nblk/ngpu)*nb;
    for(dev=0;dev<ngpu;dev++){
        // if dev < nblk%ngpu ==> this GPU has one more blk of size nb
        if(dev < nblk%ngpu){
            nloc += nb;
        }
        // this is the GPU holding the last blk
        else if (dev == nblk%ngpu){
            nloc += n%nb;
        }
        devid = (dev-first_devid+ngpu)%ngpu; 
        nlocal[devid]    = nloc;
    }


    colindex_ptr[first_devid] = 0;
    firstelement_ptr[first_devid] = 0;
    for(dev=first_devid+1;dev<ngpu+first_devid;dev++){
        devid = dev%ngpu;
        prev_devid = (dev-1+ngpu)%ngpu;        
        colindex_ptr[devid] = colindex_ptr[prev_devid] + nlocal[prev_devid]; 
        firstelement_ptr[devid] = firstelement_ptr[prev_devid] + nb; 
    }
    magma_int_t dev_locblkid  = (gbi/nb)/ngpu;
    magma_int_t dev_colindex  = dev_locblkid*nb; 






    magmaDoubleComplex *work;
    magma_queue_t stream[ngpu];
    for(dev=0;dev<ngpu;dev++){
        magma_queue_create( &stream[dev] );
    }
    magma_zmalloc_cpu( &work, max(n,m) );
    assert( work != NULL );  // TODO return error, or allocate outside zlatrd
    
    if (m >= n) {
        /* Reduce to upper bidiagonal form */
        for (i__ = 1; i__ <= nb; ++i__) {
            /*  Update A(i:m,i) */
            i__2 = m - i__ + 1;
            i__3 = i__ - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &i__3, &Y[i__+y_dim1], &ldy );
            #endif
            blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one, &A[i__ + a_dim1], &lda,
                   &Y[i__+y_dim1], &ldy, &c_one, &A[i__ + i__ * a_dim1], &c__1);
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv( &i__3, &Y[i__+y_dim1], &ldy );
            #endif
            blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one, &X[i__ + x_dim1], &ldx,
                   &A[i__*a_dim1+1], &c__1, &c_one, &A[i__+i__*a_dim1], &c__1);
            
            /* Generate reflection Q(i) to annihilate A(i+1:m,i) */
            alpha = A[i__ + i__ * a_dim1];
            i__2 = m - i__ + 1;
            i__3 = i__ + 1;
            lapackf77_zlarfg(&i__2, &alpha,
                    &A[min(i__3,m) + i__ * a_dim1], &c__1, &tauq[i__]);
            d[i__] = MAGMA_Z_REAL( alpha );
            if (i__ < n) {
                A[i__ + i__ * a_dim1] = c_one;

                /* Compute Y(i+1:n,i) */
                i__2 = m - i__ + 1;
                i__3 = n - i__; // i__3 is replaced by nlocal. note that i__3 is equal nlocal for all GPU except the first device which own the current panel where i__3 = nlocal - i__

                // 1. broadcast the block reflector  A(i+1:m,i) to the all GPU involved in computation ------
                //    transfer to dA(i,i) for the GPU owning the blockcol-panel or dwork otherwise 
                for(dev=0; dev<ngpu; dev++){
                    myn = dev==first_devid ? nlocal[dev]-(i__) : nlocal[dev];                        
                    if(myn>0){        
                        magma_setdevice(dev);
                        magma_zsetvector_async( i__2,
                              A + i__   + i__   * a_dim1, 1,
                              ( dev==first_devid ? dA[dev]+(i__-1)+ (dev_colindex+i__-1) * (ldda) : dwork[dev] ), 1, 
                              stream[dev] );
                    }
                }

                // 2. Multiply ---------------------------------------------
                // CALL DGEMV( 'Transpose', M-I+1, N-I, ONE, A( I, I+1 ), LDA, A( I, I ), 1, ZERO, Y( I+1, I ), 1 )
                // Note that every GPU compute the final GEMV results of a portion
                // of size myn of Y( I+1, I ), then on the CPU it need to be reordered 
                // to match the distribution 1D blockcolcyclic
                for(dev=0; dev<ngpu; dev++){ 
                    myn = dev==first_devid ? nlocal[dev]-(i__) : nlocal[dev];
                    if(myn > 0){
                        loc_colindex = dev==first_devid ? dev_colindex+(i__-1)+1 : dev_colindex;
                        magma_setdevice(dev);
                        magmablasSetKernelStream(stream[dev]);
                        magma_queue_sync( stream[dev] );
                        magma_zgemv(MagmaConjTrans, i__2, myn, c_one,
                                dA[dev] + (i__-1) + loc_colindex * (ldda), ldda,
                                ( dev==first_devid ? dA[dev]+(i__-1)+ (dev_colindex+i__-1) * (ldda) : dwork[dev] ), c__1, c_zero,
                                dwork2[dev], c__1);
                    }
                }
                
                // 3. Put the result back ----------------------------------
                // Since the matrix is distributed 1D col ==> each GPU compute 
                // myn portion of the final results, so gather them
                for(dev=0; dev<ngpu; dev++){
                    myn = dev==first_devid ? nlocal[dev]-(i__) : nlocal[dev];                        
                    if(myn>0){        
                        magma_setdevice(dev);
                        magma_zgetmatrix_async( myn, 1,
                                  dwork2[dev], 1,
                                  &(work[colindex_ptr[dev]]), 1, stream[dev]);
                    }
                }

                i__2 = m - i__ + 1;
                i__3 = i__ - 1;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_one, &A[i__ + a_dim1],
                              &lda, &A[i__ + i__ * a_dim1], &c__1, &c_zero,
                              &Y[i__ * y_dim1 + 1], &c__1);

                i__2 = n - i__;
                i__3 = i__ - 1;
                blasf77_zgemv("N", &i__2, &i__3, &c_neg_one, &Y[i__ + 1 +y_dim1], &ldy,
                              &Y[i__ * y_dim1 + 1], &c__1,
                              &c_zero, work, &c__1);
                i__2 = m - i__ + 1;
                i__3 = i__ - 1;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_one, &X[i__ + x_dim1],
                              &ldx, &A[i__ + i__ * a_dim1], &c__1, &c_zero,
                              &Y[i__ * y_dim1 + 1], &c__1);
                
                // 4. Synch to make sure the result is back ----------------
                for(dev=0; dev<ngpu; dev++){
                    myn = dev==first_devid ? nlocal[dev]-(i__) : nlocal[dev];                        
                    if(myn>0){        
                        magma_queue_sync( stream[dev] );
                        // reorder the received GEMV output into Y[i__+1+i__*y_dim1]
                        blkid = -1;        
                        for(k=0; k<myn; k+=ib){
                            blkid += 1;
                            ib     = (k==0 && dev==first_devid) ? nb-i__:nb; //the first block to be copied corrspond to the current panel and thus its size is nb-i__  
                            pos    = firstelement_ptr[dev] + blkid*nb*ngpu + ((k==0 && dev==first_devid) ? i__+1 : 1); //fortran indexing
                            memcpy(&(Y[pos+i__*y_dim1]), &(work[colindex_ptr[dev]+k]), min(ib,myn-k)*sizeof(magmaDoubleComplex));
                        }
                    }
                }

                if (i__3 != 0) {
                    i__2 = n - i__;
                    blasf77_zaxpy(&i__2, &c_one, work,&c__1, &Y[i__+1+i__*y_dim1],&c__1);
                }

                i__2 = i__ - 1;
                i__3 = n - i__;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_neg_one, &A[(i__ + 1) *
                              a_dim1 + 1], &lda, &Y[i__ * y_dim1 + 1], &c__1, &c_one,
                              &Y[i__ + 1 + i__ * y_dim1], &c__1);
                i__2 = n - i__;
                blasf77_zscal(&i__2, &tauq[i__], &Y[i__ + 1 + i__ * y_dim1], &c__1);

                /* Update A(i,i+1:n) */
                i__2 = n - i__;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &i__2, &A[i__+(i__+1)*a_dim1], &lda );
                lapackf77_zlacgv( &i__,  &A[i__+a_dim1], &lda );
                #endif
                blasf77_zgemv("No transpose", &i__2, &i__, &c_neg_one, &Y[i__ + 1 +
                              y_dim1], &ldy, &A[i__ + a_dim1], &lda, &c_one, &A[i__ + (
                              i__ + 1) * a_dim1], &lda);
                i__2 = i__ - 1;
                i__3 = n - i__;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &i__,  &A[i__+a_dim1], &lda );
                lapackf77_zlacgv( &i__2, &X[i__+x_dim1], &ldx );
                #endif
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_neg_one, &A[(i__ + 1) *
                              a_dim1 + 1], &lda, &X[i__ + x_dim1], &ldx, &c_one, &A[
                              i__ + (i__ + 1) * a_dim1], &lda);
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv( &i__2, &X[i__+x_dim1], &ldx );
                #endif

                /* Generate reflection P(i) to annihilate A(i,i+2:n) */
                i__2 = n - i__;
                /* Computing MIN */
                i__3 = i__ + 2;
                alpha = A[i__ + (i__ + 1) * a_dim1];
                lapackf77_zlarfg(&i__2, &alpha, &A[i__ + min(
                        i__3,n) * a_dim1], &lda, &taup[i__]);
                e[i__] = MAGMA_Z_REAL( alpha );
                A[i__ + (i__ + 1) * a_dim1] = c_one;

                /* Compute X(i+1:m,i) */
                i__2 = m - i__;
                i__3 = n - i__;
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_zsetvector( i__3,
                                  A + i__   + (i__   +1)* a_dim1, lda,
                                  dA[0]+(i__-1)+((i__-1)+1)*(ldda),  ldda );
                // 2. Multiply ---------------------------------------------
                //magma_zcopy(i__3, dA+(i__-1)+((i__-1)+1)*(ldda), ldda,
                //            dY + 1 + lddy, 1);
                magma_zgemv(MagmaNoTrans, i__2, i__3, c_one,
                            dA[0] + (i__-1)+1+ ((i__-1)+1) * (ldda), ldda,
                            dA[0] + (i__-1) +  ((i__-1)+1) * (ldda), ldda,
                            //dY + 1 + lddy, 1,
                            c_zero, dX + i__ + 1 + i__ * x_dim1, c__1);

                // 3. Put the result back ----------------------------------
                magma_zgetmatrix_async( i__2, 1,
                                        dX+i__+1+i__*x_dim1, x_dim1,
                                        X+i__+1+i__*x_dim1,  x_dim1, stream[0] );

                i__2 = n - i__;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__, &c_one, &Y[i__ + 1 + y_dim1],
                        &ldy, &A[i__ + (i__ + 1) * a_dim1], &lda, &c_zero, &X[
                        i__ * x_dim1 + 1], &c__1);

                i__2 = m - i__;
                blasf77_zgemv("N", &i__2, &i__, &c_neg_one, &A[i__ + 1 + a_dim1], &lda,
                       &X[i__ * x_dim1 + 1], &c__1, &c_zero, work, &c__1);
                i__2 = i__ - 1;
                i__3 = n - i__;
                blasf77_zgemv("N", &i__2, &i__3, &c_one, &A[(i__ + 1) * a_dim1 + 1],
                       &lda, &A[i__ + (i__ + 1) * a_dim1], &lda,
                       &c_zero, &X[i__ * x_dim1 + 1], &c__1);

                // 4. Synch to make sure the result is back ----------------
                magma_queue_sync( stream[0] );
                if (i__ != 0) {
                    i__2 = m - i__;
                    blasf77_zaxpy(&i__2, &c_one, work,&c__1, &X[i__+1+i__*x_dim1],&c__1);
                }


                i__2 = m - i__;
                i__3 = i__ - 1;
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one, &X[i__ + 1 +
                        x_dim1], &ldx, &X[i__ * x_dim1 + 1], &c__1, &c_one, &X[
                        i__ + 1 + i__ * x_dim1], &c__1);
                i__2 = m - i__;
                blasf77_zscal(&i__2, &taup[i__], &X[i__ + 1 + i__ * x_dim1], &c__1);

                #if defined(PRECISION_z) || defined(PRECISION_c)
                i__2 = n - i__;
                lapackf77_zlacgv( &i__2,  &A[i__+(i__+1)*a_dim1], &lda );
                // 4. Send the block reflector  A(i+1:m,i) to the GPU after ZLACGV()
                magma_zsetvector( i__2,
                                  A + i__   + (i__   +1)* a_dim1, lda,
                                  dA[0]+(i__-1)+((i__-1)+1)*(ldda),  ldda );
                #endif
            }
        }
    }
    else {
        /* Reduce to lower bidiagonal form */
        for (i__ = 1; i__ <= nb; ++i__) {
        
            /* Update A(i,i:n) */
            i__2 = n - i__ + 1;
            i__3 = i__ - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv(&i__2, &A[i__ + i__ * a_dim1], &lda);
            lapackf77_zlacgv(&i__3, &A[i__ + a_dim1], &lda);
            #endif
            blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one, &Y[i__ + y_dim1], &ldy,
                   &A[i__ + a_dim1], &lda, &c_one, &A[i__ + i__ * a_dim1], &lda);
            i__2 = i__ - 1;
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv(&i__3, &A[i__ + a_dim1], &lda);
            lapackf77_zlacgv(&i__3, &X[i__ + x_dim1], &ldx);
            #endif
            i__3 = n - i__ + 1;
            blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_neg_one, &A[i__ * a_dim1 + 1],
                   &lda, &X[i__ + x_dim1], &ldx, &c_one, &A[i__ + i__ * a_dim1], &lda);
            #if defined(PRECISION_z) || defined(PRECISION_c)
            lapackf77_zlacgv(&i__2, &X[i__ + x_dim1], &ldx);
            #endif
            
            /* Generate reflection P(i) to annihilate A(i,i+1:n) */
            i__2 = n - i__ + 1;
            /* Computing MIN */
            i__3 = i__ + 1;
            alpha = A[i__ + i__ * a_dim1];
            lapackf77_zlarfg(&i__2, &alpha,
                    &A[i__ + min(i__3,n) * a_dim1], &lda, &taup[i__]);
            d[i__] = MAGMA_Z_REAL( alpha );
            if (i__ < m) {
                A[i__ + i__ * a_dim1] = c_one;
                
                /* Compute X(i+1:m,i) */
                i__2 = m - i__;
                i__3 = n - i__ + 1;
                
                // 1. Send the block reflector  A(i,i+1:n) to the GPU ------
                magma_zsetvector( i__3,
                                  A + i__   + i__   * a_dim1, lda,
                                  dA[0]+(i__-1)+(i__-1)* (ldda), ldda );
                
                // 2. Multiply ---------------------------------------------
                //magma_zcopy(i__3, dA+(i__-1)+(i__-1)*(ldda), ldda,
                //            dY + 1 + lddy, 1);
                magma_zgemv(MagmaNoTrans, i__2, i__3, c_one,
                            dA[0] + (i__-1)+1 + (i__-1) * ldda, ldda,
                            dA[0] + (i__-1)   + (i__-1) * ldda, ldda,
                            // dY + 1 + lddy, 1,
                            c_zero,
                            dX + i__ + 1 + i__ * x_dim1, c__1);
                
                // 3. Put the result back ----------------------------------
                magma_zgetmatrix_async( i__2, 1,
                                        dX+i__+1+i__*x_dim1, x_dim1,
                                        X+i__+1+i__*x_dim1,  x_dim1, stream[0] );
                
                i__2 = n - i__ + 1;
                i__3 = i__ - 1;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_one, &Y[i__ + y_dim1],
                       &ldy, &A[i__ + i__ * a_dim1], &lda, &c_zero,
                       &X[i__ *  x_dim1 + 1], &c__1);
                i__2 = m - i__;
                i__3 = i__ - 1;
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one,
                              &A[i__ + 1 + a_dim1], &lda, &X[i__ * x_dim1 + 1], &c__1, &c_zero,
                              work, &c__1);
                
                i__2 = i__ - 1;
                i__3 = n - i__ + 1;
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_one,
                       &A[i__ * a_dim1 + 1], &lda, &A[i__ + i__ * a_dim1], &lda, &c_zero,
                       &X[i__ * x_dim1 + 1], &c__1);
                
                // 4. Synch to make sure the result is back ----------------
                magma_queue_sync( stream[0] );
                if (i__2 != 0) {
                    i__3 = m - i__;
                    blasf77_zaxpy(&i__3, &c_one, work,&c__1, &X[i__+1+i__*x_dim1],&c__1);
                }
                
                i__2 = m - i__;
                i__3 = i__ - 1;
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one,
                       &X[i__ + 1 + x_dim1], &ldx, &X[i__ * x_dim1 + 1], &c__1, &c_one,
                       &X[i__ + 1 + i__ * x_dim1], &c__1);
                i__2 = m - i__;
                blasf77_zscal(&i__2, &taup[i__], &X[i__ + 1 + i__ * x_dim1], &c__1);
                i__2 = n - i__ + 1;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv(&i__2, &A[i__ + i__ * a_dim1], &lda);
                magma_zsetvector( i__2,
                                  A + i__   + (i__  )* a_dim1, lda,
                                  dA[0]+(i__-1)+ (i__-1)*(ldda),  ldda );
                #endif
                
                /* Update A(i+1:m,i) */
                i__2 = m - i__;
                i__3 = i__ - 1;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv(&i__3, &Y[i__ + y_dim1], &ldy);
                #endif
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one,
                       &A[i__ + 1 + a_dim1], &lda, &Y[i__ + y_dim1], &ldy, &c_one,
                       &A[i__ + 1 + i__ * a_dim1], &c__1);
                i__2 = m - i__;
                #if defined(PRECISION_z) || defined(PRECISION_c)
                lapackf77_zlacgv(&i__3, &Y[i__ + y_dim1], &ldy);
                #endif
                blasf77_zgemv("No transpose", &i__2, &i__, &c_neg_one,
                       &X[i__ + 1 + x_dim1], &ldx, &A[i__ * a_dim1 + 1], &c__1, &c_one,
                       &A[i__ + 1 + i__ * a_dim1], &c__1);
                
                /* Generate reflection Q(i) to annihilate A(i+2:m,i) */
                i__2 = m - i__;
                i__3 = i__ + 2;
                alpha = A[i__ + 1 + i__ * a_dim1];
                lapackf77_zlarfg(&i__2, &alpha,
                        &A[min(i__3,m) + i__ * a_dim1], &c__1, &tauq[i__]);
                e[i__] = MAGMA_Z_REAL( alpha );
                A[i__ + 1 + i__ * a_dim1] = c_one;
                
                /* Compute Y(i+1:n,i) */
                i__2 = m - i__;
                i__3 = n - i__;
                
                // 1. Send the block reflector  A(i+1:m,i) to the GPU ------
                magma_zsetvector( i__2,
                                  A + i__   +1+  i__   * a_dim1, 1,
                                  dA[0]+(i__-1)+1+ (i__-1)*(ldda),  1 );
                // 2. Multiply ---------------------------------------------
                magma_zgemv(MagmaConjTrans, i__2, i__3, c_one,
                            dA[0] + (i__-1)+1+ ((i__-1)+1) * ldda, ldda,
                            dA[0] + (i__-1)+1+  (i__-1)    * ldda, c__1,
                            c_zero, dY + i__ + 1 + i__ * y_dim1, c__1);
                
                // 3. Put the result back ----------------------------------
                magma_zgetmatrix_async( i__3, 1,
                                        dY+i__+1+i__*y_dim1, y_dim1,
                                        Y+i__+1+i__*y_dim1,  y_dim1, stream[0] );
                
                i__2 = m - i__;
                i__3 = i__ - 1;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__3, &c_one, &A[i__ + 1 + a_dim1],
                       &lda, &A[i__ + 1 + i__ * a_dim1], &c__1, &c_zero,
                       &Y[ i__ * y_dim1 + 1], &c__1);
                i__2 = n - i__;
                i__3 = i__ - 1;
                blasf77_zgemv("No transpose", &i__2, &i__3, &c_neg_one,
                       &Y[i__ + 1 + y_dim1], &ldy, &Y[i__ * y_dim1 + 1], &c__1,
                       &c_zero, work, &c__1);
                
                i__2 = m - i__;
                blasf77_zgemv(MagmaConjTransStr, &i__2, &i__, &c_one, &X[i__ + 1 + x_dim1],
                       &ldx, &A[i__ + 1 + i__ * a_dim1], &c__1, &c_zero,
                       &Y[i__ * y_dim1 + 1], &c__1);
                
                // 4. Synch to make sure the result is back ----------------
                magma_queue_sync( stream[0] );
                if (i__3 != 0) {
                    i__2 = n - i__;
                    blasf77_zaxpy(&i__2, &c_one, work,&c__1, &Y[i__+1+i__*y_dim1],&c__1);
                }
                
                i__2 = n - i__;
                blasf77_zgemv(MagmaConjTransStr, &i__, &i__2, &c_neg_one,
                       &A[(i__ + 1) * a_dim1 + 1], &lda, &Y[i__ * y_dim1 + 1],
                       &c__1, &c_one, &Y[i__ + 1 + i__ * y_dim1], &c__1);
                i__2 = n - i__;
                blasf77_zscal(&i__2, &tauq[i__], &Y[i__ + 1 + i__ * y_dim1], &c__1);
            }
            #if defined(PRECISION_z) || defined(PRECISION_c)
            else {
                i__2 = n - i__ + 1;
                lapackf77_zlacgv(&i__2, &A[i__ + i__ * a_dim1], &lda);
                magma_zsetvector( i__2,
                                  A + i__   + (i__  )* a_dim1, lda,
                                  dA[0]+(i__-1)+ (i__-1)*(ldda),  ldda );
            }
            #endif
        }
    }
    
    for(dev=0;dev<ngpu;dev++){
        magma_queue_destroy( stream[dev] );
    }
    magma_free_cpu(work);
    
    return MAGMA_SUCCESS;
} /* magma_zlabrd_gpu */
