/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Stan Tomov
       @author Mark Gates
*/
#include "common_magma.h"

#define PRECISION_z

extern "C" magma_int_t
magma_zlahr2(
    magma_int_t n, magma_int_t k, magma_int_t nb,
    cuDoubleComplex *dA, cuDoubleComplex *dV,
    cuDoubleComplex *A, magma_int_t lda,
    cuDoubleComplex *tau,
    cuDoubleComplex *T, magma_int_t ldt,
    cuDoubleComplex *Y, magma_int_t ldy )
{
/*  -- MAGMA auxiliary routine (version 1.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose   
    =======   

    ZLAHR2 reduces the first NB columns of a complex general n-BY-(n-k+1)   
    matrix A so that elements below the k-th subdiagonal are zero. The   
    reduction is performed by an orthogonal similarity transformation   
    Q' * A * Q. The routine returns the matrices V and T which determine   
    Q as a block reflector I - V*T*V', and also the matrix Y = A * V.   
    (Note this is different than LAPACK, which computes Y = A * V * T.)

    This is an auxiliary routine called by ZGEHRD.   

    Arguments   
    =========   

    N       (input) INTEGER   
            The order of the matrix A.   

    K       (input) INTEGER   
            The offset for the reduction. Elements below the k-th   
            subdiagonal in the first NB columns are reduced to zero.   
            K < N.   

    NB      (input) INTEGER   
            The number of columns to be reduced.

    dA      (input/output) COMPLEX_16 array on the GPU, dimension (LDA,N-K+1)   
            On entry, the n-by-(n-k+1) general matrix A.
            On exit, the elements in rows K:N of the first NB columns are
            overwritten with the matrix Y.

    DV      (output) COMPLEX_16 array on the GPU, dimension (N, NB)
            On exit this contains the Householder vectors of the transformation.

    A       (input/output) COMPLEX_16 array, dimension (LDA,N-K+1)   
            On entry, the n-by-(n-k+1) general matrix A.   
            On exit, the elements on and above the k-th subdiagonal in   
            the first NB columns are overwritten with the corresponding   
            elements of the reduced matrix; the elements below the k-th   
            subdiagonal, with the array TAU, represent the matrix Q as a   
            product of elementary reflectors. The other columns of A are   
            unchanged. See Further Details.   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,N).   

    TAU     (output) COMPLEX_16 array, dimension (NB)   
            The scalar factors of the elementary reflectors. See Further   
            Details.   

    T       (output) COMPLEX_16 array, dimension (LDT,NB)   
            The upper triangular matrix T.   

    LDT     (input) INTEGER   
            The leading dimension of the array T.  LDT >= NB.   

    Y       (output) COMPLEX_16 array, dimension (LDY,NB)   
            The n-by-nb matrix Y.   

    LDY     (input) INTEGER   
            The leading dimension of the array Y. LDY >= N.   

    Further Details   
    ===============   
    The matrix Q is represented as a product of nb elementary reflectors   

       Q = H(1) H(2) . . . H(nb).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a complex scalar, and v is a complex vector with   
    v(1:i+k-1) = 0, v(i+k) = 1; v(i+k+1:n) is stored on exit in   
    A(i+k+1:n,i), and tau in TAU(i).   

    The elements of the vectors v together form the (n-k+1)-by-nb matrix   
    V which is needed, with T and Y, to apply the transformation to the   
    unreduced part of the matrix, using an update of the form:   
    A := (I - V*T*V') * (A - Y*T*V').   

    The contents of A on exit are illustrated by the following example   
    with n = 7, k = 3 and nb = 2:   

       ( a   a   a   a   a )   
       ( a   a   a   a   a )   
       ( a   a   a   a   a )   
       ( h   h   a   a   a )   
       ( v1  h   a   a   a )   
       ( v1  v2  a   a   a )   
       ( v1  v2  a   a   a )   

    where "a" denotes an element of the original matrix A, h denotes a
    modified element of the upper Hessenberg matrix H, and vi denotes an   
    element of the vector defining H(i).

    This implementation follows the hybrid algorithm and notations described in

    S. Tomov and J. Dongarra, "Accelerating the reduction to upper Hessenberg
    form through hybrid GPU-based computing," University of Tennessee Computer
    Science Technical Report, UT-CS-09-642 (also LAPACK Working Note 219),
    May 24, 2009.
    =====================================================================    */

    #define  A( i, j ) ( A + (i) + (j)*lda)
    #define  Y( i, j ) ( Y + (i) + (j)*ldy)
    #define  T( i, j ) ( T + (i) + (j)*ldt)
    #define dA( i, j ) (dA + (i) + (j)*ldda)
    #define dV( i, j ) (dV + (i) + (j)*ldda)
    
    cuDoubleComplex c_zero    = MAGMA_Z_ZERO;
    cuDoubleComplex c_one     = MAGMA_Z_ONE;
    cuDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t ldda = lda;
    magma_int_t ione = 1;
    
    magma_int_t n_k_i_1, n_k;
    cuDoubleComplex scale;

    magma_int_t i;
    cuDoubleComplex ei;

    // adjust from 1-based indexing
    k -= 1;

    // Function Body
    if (n <= 1)
        return 0;
    
    for (i = 0; i < nb; ++i) {
        n_k_i_1 = n - k - i - 1;
        n_k     = n - k;
        
        if (i > 0) {
            // Update A(k:n-1,i); Update i-th column of A - Y * T * V'
            // This updates one more row than LAPACK does (row k),
            // making the block above the panel an even multiple of nb.
            // Use last column of T as workspace, w.
            // w(0:i-1, nb-1) = VA(k+i, 0:i-1)'
            blasf77_zcopy( &i,
                           A(k+i,0),  &lda,
                           T(0,nb-1), &ione );
            #if defined(PRECISION_z) || defined(PRECISION_c)
            // If complex, conjugate row of V.
            lapackf77_zlacgv(&i, T(0,nb-1), &ione);
            #endif
            
            // w = T(0:i-1, 0:i-1) * w
            blasf77_ztrmv( "Upper", "No trans", "No trans", &i,
                           T(0,0),    &ldt,
                           T(0,nb-1), &ione );
            
            // A(k:n-1, i) -= Y(k:n-1, 0:i-1) * w
            blasf77_zgemv( "No trans", &n_k, &i,
                           &c_neg_one, Y(k,0),    &ldy,
                                       T(0,nb-1), &ione,
                           &c_one,     A(k,i),    &ione );
            
            // Apply I - V * T' * V' to this column (call it b) from the
            // left, using the last column of T as workspace, w.
            //
            // Let  V = ( V1 )   and   b = ( b1 )   (first i-1 rows)
            //          ( V2 )             ( b2 )
            // where V1 is unit lower triangular
            
            // w := b1 = A(k+1:k+i, i)
            blasf77_zcopy( &i,
                           A(k+1,i),  &ione,
                           T(0,nb-1), &ione );
            
            // w := V1' * b1 = VA(k+1:k+i, 0:i-1)' * w
            blasf77_ztrmv( "Lower", "Conj", "Unit", &i,
                           A(k+1,0), &lda,
                           T(0,nb-1), &ione );
            
            // w := w + V2'*b2 = w + VA(k+i+1:n-1, 0:i-1)' * A(k+i+1:n-1, i)
            blasf77_zgemv( "Conj", &n_k_i_1, &i,
                           &c_one, A(k+i+1,0), &lda,
                                   A(k+i+1,i), &ione,
                           &c_one, T(0,nb-1),  &ione );
            
            // w := T'*w = T(0:i-1, 0:i-1)' * w
            blasf77_ztrmv( "Upper", "Conj", "Non-unit", &i,
                           T(0,0), &ldt,
                           T(0,nb-1), &ione );
            
            // b2 := b2 - V2*w = A(k+i+1:n-1, i) - VA(k+i+1:n-1, 0:i-1) * w
            blasf77_zgemv( "No trans", &n_k_i_1, &i,
                           &c_neg_one, A(k+i+1,0), &lda,
                                       T(0,nb-1),  &ione,
                           &c_one,     A(k+i+1,i), &ione );
            
            // w := V1*w = VA(k+1:k+i, 0:i-1) * w
            blasf77_ztrmv( "Lower", "No trans", "Unit", &i,
                           A(k+1,0), &lda,
                           T(0,nb-1), &ione );
            
            // b1 := b1 - w = A(k+1:k+i-1, i) - w
            blasf77_zaxpy( &i,
                           &c_neg_one, T(0,nb-1), &ione,
                                       A(k+1,i),    &ione );
            
            // Restore diagonal element, saved below during previous iteration
            *A(k+i,i-1) = ei;
        }
        
        // Generate the elementary reflector H(i) to annihilate A(k+i+1:n-1,i)
        lapackf77_zlarfg( &n_k_i_1,
                          A(k+i+1,i),
                          A(k+i+2,i), &ione, &tau[i] );
        // Save diagonal element and set to one, to simplify multiplying by V
        ei = *A(k+i+1,i);
        *A(k+i+1,i) = c_one;

        // dV(i+1:n-k-1, i) = VA(k+i+1:n-1, i)
        magma_zsetvector( n_k_i_1,
                          A(k+i+1,i), 1,
                          dV(i+1,i),  1 );
        
        // Compute Y(k+1:n,i) = A vi
        // dA(k:n-1, i) = dA(k:n-1, i+1:n-k-1) * dV(i+1:n-k-1, i)
        magma_zgemv( MagmaNoTrans, n_k, n_k_i_1,
                     c_one,  dA(k,i+1), ldda,
                             dV(i+1,i),   ione,
                     c_zero, dA(k,i),     ione );
        
        // Compute T(0:i,i) = [ -tau T V' vi ]
        //                    [  tau         ]
        // T(0:i-1, i) = -tau VA(k+i+1:n-1, 0:i-1)' VA(k+i+1:n-1, i)
        scale = MAGMA_Z_NEGATE( tau[i]);
        blasf77_zgemv( "Conj", &n_k_i_1, &i,
                       &scale,  A(k+i+1,0), &lda,
                                A(k+i+1,i), &ione,
                       &c_zero, T(0,i),     &ione );
        // T(0:i-1, i) = T(0:i-1, 0:i-1) * T(0:i-1, i)
        blasf77_ztrmv( "Upper", "No trans", "Non-unit", &i,
                       T(0,0), &ldt,
                       T(0,i), &ione );
        *T(i,i) = tau[i];

        // Y(k:n-1, i) = dA(k:n-1, i)
        magma_zgetvector( n-k,
                          dA(k,i), 1,
                          Y(k,i),  1 );
    }
    // Restore diagonal element
    *A(k+nb,nb-1) = ei;

    return 0;
} // magma_zlahr2
