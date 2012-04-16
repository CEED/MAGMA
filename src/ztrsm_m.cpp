/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Raffaele Solca

       @precisions normal z -> s d c
*/
#define N_MAX_GPU 8

#include "common_magma.h"

extern "C"
magma_int_t magma_get_ztrsm_m_nb() { return 128;}

#define A(i, j) (a+(j)*nb*lda + (i)*nb)
#define B(i, j) (b+(j)*nb*ldb + (i)*nb)

#define dB(gpui, i, j) (dw[gpui] + (j)*nb*lddb + (i)*nb)

#define dA(gpui, i, j) (dw[gpui] + dimb*lddb + (i)*nb + (j)*nb*ldda)

extern "C" magma_int_t
magma_ztrsm_m (magma_int_t nrgpu, char side, char uplo, char transa, char diag,
         magma_int_t m, magma_int_t n, cuDoubleComplex alpha, cuDoubleComplex *a,
         magma_int_t lda, cuDoubleComplex *b, magma_int_t ldb)
{

/*  Purpose
    =======

    ZTRSM  solves one of the matrix equations

       op( A )*X = alpha*B,   or   X*op( A ) = alpha*B,

    where alpha is a scalar, X and B are m by n matrices, A is a unit, or

    non-unit,  upper or lower triangular matrix  and  op( A )  is one  of


       op( A ) = A   or   op( A ) = A'   or   op( A ) = conjg( A' ).

    The matrix X is overwritten on B.

    Parameters
    ==========

    SIDE   - CHARACTER*1.
             On entry, SIDE specifies whether op( A ) appears on the left

             or right of X as follows:

                SIDE = 'L' or 'l'   op( A )*X = alpha*B.

                SIDE = 'R' or 'r'   X*op( A ) = alpha*B.

             Unchanged on exit.

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the matrix A is an upper or

             lower triangular matrix as follows:

                UPLO = 'U' or 'u'   A is an upper triangular matrix.

                UPLO = 'L' or 'l'   A is a lower triangular matrix.

             Unchanged on exit.

    TRANSA - CHARACTER*1.
             On entry, TRANSA specifies the form of op( A ) to be used in

             the matrix multiplication as follows:

                TRANSA = 'N' or 'n'   op( A ) = A.

                TRANSA = 'T' or 't'   op( A ) = A'.

                TRANSA = 'C' or 'c'   op( A ) = conjg( A' ).

             Unchanged on exit.

    DIAG   - CHARACTER*1.
             On entry, DIAG specifies whether or not A is unit triangular

             as follows:

                DIAG = 'U' or 'u'   A is assumed to be unit triangular.

                DIAG = 'N' or 'n'   A is not assumed to be unit
                                    triangular.

             Unchanged on exit.

    M      - INTEGER.
             On entry, M specifies the number of rows of B. M must be at

             least zero.
             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the number of columns of B.  N must be

             at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry,  ALPHA specifies the scalar  alpha. When  alpha is

             zero then  A is not referenced and  B need not be set before

             entry.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, k ), where k is m

             when  SIDE = 'L' or 'l'  and is  n  when  SIDE = 'R' or 'r'.

             Before entry  with  UPLO = 'U' or 'u',  the  leading  k by k

             upper triangular part of the array  A must contain the upper

             triangular matrix  and the strictly lower triangular part of

             A is not referenced.
             Before entry  with  UPLO = 'L' or 'l',  the  leading  k by k

             lower triangular part of the array  A must contain the lower

             triangular matrix  and the strictly upper triangular part of

             A is not referenced.
             Note that when  DIAG = 'U' or 'u',  the diagonal elements of

             A  are not referenced either,  but are assumed to be  unity.

             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared

             in the calling (sub) program.  When  SIDE = 'L' or 'l'  then

             LDA  must be at least  max( 1, m ),  when  SIDE = 'R' or 'r'

             then LDA must be at least max( 1, n ).
             Unchanged on exit.

    B      - COMPLEX*16       array of DIMENSION ( LDB, n ).
             Before entry,  the leading  m by n part of the array  B must

             contain  the  right-hand  side  matrix  B,  and  on exit  is

             overwritten by the solution matrix  X.

    LDB    - INTEGER.
             On entry, LDB specifies the first dimension of B as declared

             in  the  calling  (sub)  program.   LDB  must  be  at  least

             max( 1, m ).
             Unchanged on exit.*/

    char side_[2] = {side, 0};
    char uplo_[2] = {uplo, 0};
    char transa_[2] = {transa, 0};
    char diag_[2] = {diag, 0};
    cuDoubleComplex  zone  = MAGMA_Z_ONE;
    cuDoubleComplex  mzone = MAGMA_Z_NEG_ONE;
    cuDoubleComplex  alpha_;
    cuDoubleComplex* dw[N_MAX_GPU];
    cudaStream_t stream [N_MAX_GPU][3];
    magma_int_t lside;
    magma_int_t upper;
    magma_int_t notransp;
    magma_int_t nrowa;
    magma_int_t nb = magma_get_ztrsm_m_nb();
    magma_int_t igpu = 0;
    magma_int_t info;
    magma_int_t k,j,jj,kb,jb,jjb;
    magma_int_t ldda, dima, lddb, dimb;
    int gpu_b;
    cudaGetDevice(&gpu_b);

    lside = lapackf77_lsame(side_, "L");
    if (lside) {
        nrowa = m;
    } else {
        nrowa = n;
    }
    upper = lapackf77_lsame(uplo_, "U");
    notransp = lapackf77_lsame(transa_, "N");

    info = 0;
    if (! lside && ! lapackf77_lsame(side_, "R")) {
        info = 1;
    } else if (! upper && ! lapackf77_lsame(uplo_, "L")) {
        info = 2;
    } else if (! notransp && ! lapackf77_lsame(transa_, "T")
               && ! lapackf77_lsame(transa_, "C")) {
        info = 3;
    } else if (! lapackf77_lsame(diag_, "U") && ! lapackf77_lsame(diag_, "N")) {
        info = 4;
    } else if (m < 0) {
        info = 5;
    } else if (n < 0) {
        info = 6;
    } else if (lda < max(1,nrowa)) {
        info = 9;
    } else if (ldb < max(1,m)) {
        info = 11;
    }

    if (info != 0) {
        magma_xerbla( __func__, -info );
        return info;
    }

    //Quick return if possible.

    if (n == 0) {
        return info;
    }

    magma_int_t nbl = (n-1)/nb+1; // number of blocks in a row
    magma_int_t mbl = (m-1)/nb+1; // number of blocks in a column

    if (lside) {
        lddb = m;
        dimb = ((nbl-1)/nrgpu+1)*nb;
        if ( notransp ) {
            ldda = m;
            dima = 2 * nb;
        } else {
            ldda = 2 * nb;
            dima = m;
        }
    } else {
        lddb = ((mbl-1)/nrgpu+1)*nb;
        dimb = n;
        if ( !notransp ) {
            ldda = n;
            dima = 2 * nb;
        } else {
            ldda = 2 * nb;
            dima = n;
        }
    }

    for (igpu = 0; igpu < nrgpu; ++igpu){
        cudaSetDevice(igpu);
        if (cudaSuccess != cudaMalloc( (void**)&dw[igpu], (dimb*lddb + dima*ldda)*sizeof(cuDoubleComplex) ) ) {
            info = MAGMA_ERR_CUBLASALLOC;
            return info;
        }
        cudaStreamCreate(&stream[igpu][0]);
        cudaStreamCreate(&stream[igpu][1]);
        cudaStreamCreate(&stream[igpu][2]);
    }

    // alpha = 0 case;

    if (MAGMA_Z_REAL(alpha) == 0. && MAGMA_Z_IMAG(alpha) == 0.) {
        printf("ztrsm_m: alpha = 0 not implemented\n");
        exit(-1);

        return info;
    }

    if (lside) {
        if (notransp) {

            //Form  B := alpha*inv( A )*B

            if (upper) {

                //left upper notranspose

                magma_int_t nloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    nloc[igpu] = 0;

                //copy B to mgpus
                for (k = 0; k < nbl; ++k){
                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    kb = min(nb, n-k*nb);
                    nloc[igpu] += kb;
                    cudaMemcpy2DAsync(dB(igpu, 0, k/nrgpu), lddb * sizeof(cuDoubleComplex),
                                      B(0, k), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][(mbl+1)%2]);
                }
                jb = min(nb, m-(mbl-1)*nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, (mbl-1)%2), ldda * sizeof(cuDoubleComplex),
                                      A(0, mbl-1), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, jb,
                                      cudaMemcpyHostToDevice, stream[igpu][(mbl+1)%2]);
                }
                for (j = mbl-1; j >= 0; --j){
                    if (j > 0){
                        jb = nb;
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, 0, (j+1)%2), ldda * sizeof(cuDoubleComplex),
                                              A(0, j-1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*j*nb, jb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    if (j==mbl-1)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    jb = min(nb, m-j*nb);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm(side, uplo, transa, diag, jb, nloc[igpu], alpha_, dA(igpu, j, j%2), ldda,
                                    dB(igpu, j, 0), lddb );
                    }

                    if (j>0){
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm(transa, 'N', j*nb, nloc[igpu], jb, mzone, dA(igpu, 0, j%2), ldda,
                                        dB(igpu, j, 0), lddb, alpha_, dB(igpu, 0, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < nbl; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j, k/nrgpu), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }

            }
            else
            {
                //left lower notranspose

                magma_int_t nloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    nloc[igpu] = 0;

                //copy B to mgpus
                for (k = 0; k < nbl; ++k){
                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    kb = min(nb, n-k*nb);
                    nloc[igpu] += kb;
                    cudaMemcpy2DAsync(dB(igpu, 0, k/nrgpu), lddb * sizeof(cuDoubleComplex),
                                      B(0, k), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                jb = min(nb, m);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, 0), ldda * sizeof(cuDoubleComplex),
                                      A(0, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, jb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                for (j = 0; j < mbl; ++j){
                    if ((j+1)*nb < m){
                        jb = min(nb, m-(j+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, j+1, (j+1)%2), ldda * sizeof(cuDoubleComplex),
                                              A(j+1, j+1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*(m-(j+1)*nb), jb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    jb = min(nb, m-j*nb);

                    if (j==0)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm(side, uplo, transa, diag, jb, nloc[igpu], alpha_, dA(igpu, j, j%2), ldda,
                                    dB(igpu, j, 0), lddb );
                    }

                    if ( j < mbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm(transa, 'N', m-(j+1)*nb, nloc[igpu], nb, mzone, dA(igpu, j+1, j%2), ldda,
                                        dB(igpu, j, 0), lddb, alpha_, dB(igpu, j+1, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < nbl; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j, k/nrgpu), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }

            }
        }
        else
        {

            //Form  B := alpha*inv( A' )*B

            if (upper) {

                //left upper transpose or conjtranspose

                magma_int_t nloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    nloc[igpu] = 0;

                //copy B to mgpus
                for (k = 0; k < nbl; ++k){
                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    kb = min(nb, n-k*nb);
                    nloc[igpu] += kb;
                    cudaMemcpy2DAsync(dB(igpu, 0, k/nrgpu), lddb * sizeof(cuDoubleComplex),
                                      B(0, k), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                jb = min(nb, m);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, 0), ldda * sizeof(cuDoubleComplex),
                                      A(0, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, m,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                for (j = 0; j < mbl; ++j){
                    if ((j+1)*nb < m){
                        jb = min(nb, m-(j+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, (j+1)%2, j+1), ldda * sizeof(cuDoubleComplex),
                                              A(j+1, j+1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*jb, m-(j+1)*nb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    jb = min(nb, m-j*nb);

                    if (j==0)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm(side, uplo, transa, diag, jb, nloc[igpu], alpha_, dA(igpu, j%2, j), ldda,
                                    dB(igpu, j, 0), lddb );
                    }

                    if ( j < mbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm(transa, 'N', m-(j+1)*nb, nloc[igpu], nb, mzone, dA(igpu, j%2, j+1), ldda,
                                        dB(igpu, j, 0), lddb, alpha_, dB(igpu, j+1, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < nbl; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j, k/nrgpu), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }
            else
            {

                //left lower transpose or conjtranspose

                magma_int_t nloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    nloc[igpu] = 0;

                //copy B to mgpus
                for (k = 0; k < nbl; ++k){
                    igpu = k%nrgpu;
                    cudaSetDevice(igpu);
                    kb = min(nb, n-k*nb);
                    nloc[igpu] += kb;
                    cudaMemcpy2DAsync(dB(igpu, 0, k/nrgpu), lddb * sizeof(cuDoubleComplex),
                                      B(0, k), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*m, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][(mbl+1)%2]);
                }
                jb = min(nb, m-(mbl-1)*nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, (mbl-1)%2, 0), ldda * sizeof(cuDoubleComplex),
                                      A(mbl-1, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, m,
                                      cudaMemcpyHostToDevice, stream[igpu][(mbl+1)%2]);
                }
                for (j = mbl-1; j >= 0; --j){
                    if (j > 0){
                        jb = nb;
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, (j+1)%2, 0), ldda * sizeof(cuDoubleComplex),
                                              A(j-1, 0), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*jb, j*nb,
                                              cudaMemcpyHostToDevice, stream[igpu][(j+1)%2]);
                        }
                    }
                    if (j==mbl-1)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    jb = min(nb, m-j*nb);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][j%2]);
                        cublasZtrsm(side, uplo, transa, diag, jb, nloc[igpu], alpha_, dA(igpu, j%2, j), ldda,
                                    dB(igpu, j, 0), lddb );
                    }

                    if (j>0){
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][j%2]);
                            cublasZgemm(transa, 'N', j*nb, nloc[igpu], jb, mzone, dA(igpu, j%2, 0), ldda,
                                        dB(igpu, j, 0), lddb, alpha_, dB(igpu, 0, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][j%2]);
                    }

                    for (k = 0; k < nbl; ++k){
                        igpu = k%nrgpu;
                        cudaSetDevice(igpu);
                        kb = min(nb, n-k*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j, k/nrgpu), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }

            }
        }
    }
    else
    {
        if (notransp) {

            //Form  B := alpha*B*inv( A ).

            if (upper) {

                //right upper notranspose
                magma_int_t mloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    mloc[igpu] = 0;

                //copy B to mgpus
                for (j = 0; j < mbl; ++j){
                    igpu = j%nrgpu;
                    cudaSetDevice(igpu);
                    jb = min(nb, m-j*nb);
                    mloc[igpu] += jb;
                    cudaMemcpy2DAsync(dB(igpu, j/nrgpu, 0), lddb * sizeof(cuDoubleComplex),
                                      B(j, 0), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                kb = min(nb, n);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, 0), ldda * sizeof(cuDoubleComplex),
                                      A(0, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*kb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                for (k = 0; k < nbl; ++k){
                    if ((k+1)*nb < n){
                        kb = min(nb, n-(k+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, (k+1)%2, k+1), ldda * sizeof(cuDoubleComplex),
                                              A(k+1, k+1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb, n-(k+1)*nb,
                                              cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                        }
                    }
                    kb = min(nb, n-k*nb);

                    if (k==0)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][k%2]);
                        cublasZtrsm(side, uplo, transa, diag, mloc[igpu], kb, alpha_, dA(igpu, k%2, k), ldda,
                                    dB(igpu, 0, k), lddb );
                    }

                    if ( k < nbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][k%2]);
                            cublasZgemm('N', transa, mloc[igpu], n-(k+1)*nb, nb, mzone, dB(igpu, 0, k), lddb,
                                        dA(igpu, k%2, k+1), ldda, alpha_, dB(igpu, 0, k+1), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][k%2]);
                    }

                    for (j = 0; j < mbl; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        jb = min(nb, m-j*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j/nrgpu, k), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }
            else
            {

                //right lower notranspose
                magma_int_t mloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    mloc[igpu] = 0;

                //copy B to mgpus
                for (j = 0; j < mbl; ++j){
                    igpu = j%nrgpu;
                    cudaSetDevice(igpu);
                    jb = min(nb, m-j*nb);
                    mloc[igpu] += jb;
                    cudaMemcpy2DAsync(dB(igpu, j/nrgpu, 0), lddb * sizeof(cuDoubleComplex),
                                      B(j, 0), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][(nbl+1)%2]);
                }
                kb = min(nb, n-(nbl-1)*nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, (nbl-1)%2, 0), ldda * sizeof(cuDoubleComplex),
                                      A(nbl-1, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*kb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][(nbl+1)%2]);
                }
                for (k = nbl-1; k >= 0; --k){
                    if (k > 0){
                        kb = nb;
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, (k+1)%2, 0), ldda * sizeof(cuDoubleComplex),
                                              A(k-1, 0), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*kb, k*nb,
                                              cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                        }
                    }
                    if (k==nbl-1)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    kb = min(nb, n-k*nb);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][k%2]);
                        cublasZtrsm(side, uplo, transa, diag, mloc[igpu], kb, alpha_, dA(igpu, k%2, k), ldda,
                                    dB(igpu, 0, k), lddb );
                    }

                    if (k>0){
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][k%2]);
                            cublasZgemm('N', transa, mloc[igpu], k*nb, kb, mzone, dB(igpu, 0, k), lddb,
                                        dA(igpu, k%2, 0), ldda, alpha_, dB(igpu, 0, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][k%2]);
                    }

                    for (j = 0; j < mbl; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        jb = min(nb, m-j*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j/nrgpu, k), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }
        }
        else
        {

            //Form  B := alpha*B*inv( A' ).

            if (upper) {

                //right upper transpose or conjtranspose
                magma_int_t mloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    mloc[igpu] = 0;

                //copy B to mgpus
                for (j = 0; j < mbl; ++j){
                    igpu = j%nrgpu;
                    cudaSetDevice(igpu);
                    jb = min(nb, m-j*nb);
                    mloc[igpu] += jb;
                    cudaMemcpy2DAsync(dB(igpu, j/nrgpu, 0), lddb * sizeof(cuDoubleComplex),
                                      B(j, 0), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][(nbl+1)%2]);
                }
                kb = min(nb, n-(nbl-1)*nb);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, (nbl-1)%2), ldda * sizeof(cuDoubleComplex),
                                      A(0, nbl-1), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*n, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][(nbl+1)%2]);
                }
                for (k = nbl-1; k >= 0; --k){
                    if (k > 0){
                        kb = nb;
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, 0, (k+1)%2), ldda * sizeof(cuDoubleComplex),
                                              A(0, k-1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*k*nb, kb,
                                              cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                        }
                    }
                    if (k==nbl-1)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    kb = min(nb, n-k*nb);

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][k%2]);
                        cublasZtrsm(side, uplo, transa, diag, mloc[igpu], kb, alpha_, dA(igpu, k, k%2), ldda,
                                    dB(igpu, 0, k), lddb );
                    }

                    if (k>0){
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][k%2]);
                            cublasZgemm('N', transa, mloc[igpu], k*nb, kb, mzone, dB(igpu, 0, k), lddb,
                                        dA(igpu, 0, k%2), ldda, alpha_, dB(igpu, 0, 0), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][k%2]);
                    }

                    for (j = 0; j < mbl; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        jb = min(nb, m-j*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j/nrgpu, k), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }
            else
            {

                //right lower transpose or conjtranspose
                magma_int_t mloc[N_MAX_GPU];
                for(igpu = 0; igpu < nrgpu; ++igpu)
                    mloc[igpu] = 0;

                //copy B to mgpus
                for (j = 0; j < mbl; ++j){
                    igpu = j%nrgpu;
                    cudaSetDevice(igpu);
                    jb = min(nb, m-j*nb);
                    mloc[igpu] += jb;
                    cudaMemcpy2DAsync(dB(igpu, j/nrgpu, 0), lddb * sizeof(cuDoubleComplex),
                                      B(j, 0), ldb  * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*jb, n,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                kb = min(nb, n);
                for (igpu = 0; igpu < nrgpu; ++igpu){
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dA(igpu, 0, 0), ldda * sizeof(cuDoubleComplex),
                                      A(0, 0), lda * sizeof(cuDoubleComplex),
                                      sizeof(cuDoubleComplex)*n, kb,
                                      cudaMemcpyHostToDevice, stream[igpu][0]);
                }
                for (k = 0; k < nbl; ++k){
                    if ((k+1)*nb < n){
                        kb = min(nb, n-(k+1)*nb);
                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dA(igpu, k+1, (k+1)%2), ldda * sizeof(cuDoubleComplex),
                                              A(k+1, k+1), lda * sizeof(cuDoubleComplex),
                                              sizeof(cuDoubleComplex)*(n-(k+1)*nb), kb,
                                              cudaMemcpyHostToDevice, stream[igpu][(k+1)%2]);
                        }
                    }
                    kb = min(nb, n-k*nb);

                    if (k==0)
                        alpha_=alpha;
                    else
                        alpha_= zone;

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][k%2]);
                        cublasZtrsm(side, uplo, transa, diag, mloc[igpu], kb, alpha_, dA(igpu, k, k%2), ldda,
                                    dB(igpu, 0, k), lddb );
                    }

                    if ( k < nbl-1 ){

                        for (igpu = 0; igpu < nrgpu; ++igpu){
                            cudaSetDevice(igpu);
                            cublasSetKernelStream(stream[igpu][k%2]);
                            cublasZgemm('N', transa, mloc[igpu], n-(k+1)*nb, nb, mzone, dB(igpu, 0, k), lddb,
                                        dA(igpu, k+1, k%2), ldda, alpha_, dB(igpu, 0, k+1), lddb );
                        }
                    }

                    for (igpu = 0; igpu < nrgpu; ++igpu){
                        cudaStreamSynchronize(stream[igpu][k%2]);
                    }

                    for (j = 0; j < mbl; ++j){
                        igpu = j%nrgpu;
                        cudaSetDevice(igpu);
                        jb = min(nb, m-j*nb);
                        cudaMemcpy2DAsync(B(j, k), ldb * sizeof(cuDoubleComplex),
                                          dB(igpu, j/nrgpu, k), lddb  * sizeof(cuDoubleComplex),
                                          sizeof(cuDoubleComplex)*jb, kb,
                                          cudaMemcpyDeviceToHost, stream[igpu][2]);
                    }
                }
            }
        }

    }


    for (igpu = 0; igpu < nrgpu; ++igpu){
        cudaSetDevice(igpu);
        cublasSetKernelStream(NULL);
        cudaStreamSynchronize(stream[igpu][2]);
        cudaStreamDestroy(stream[igpu][0]);
        cudaStreamDestroy(stream[igpu][1]);
        cudaStreamDestroy(stream[igpu][2]);
        cudaFree(dw[igpu]);
    }

    cudaSetDevice(gpu_b);

    return info;

} /* magma_ztrsm_m */


