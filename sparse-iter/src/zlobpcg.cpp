/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver

       November 2011
            
       @author Stan Tomov

       @precisions normal z -> s d c
*/

#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"
#include "trace.h"
     
//==================================================================================
extern "C" magma_int_t
magma_zcompact(magma_int_t m, magma_int_t n,
               magmaDoubleComplex *dA, magma_int_t ldda,
               double *dnorms, double tol, 
               magma_int_t *activeMask, magma_int_t *cBlockSize);

extern "C" magma_int_t
magma_zcompactActive(magma_int_t m, magma_int_t n,
                     magmaDoubleComplex *dA, magma_int_t ldda, 
                     magma_int_t *active);
//==================================================================================

#define PRECISION_z


magma_int_t
magma_zlobpcg( magma_int_t m, magma_int_t n, magma_z_sparse_matrix A, 
               magmaDoubleComplex *blockX, double *evalues,
               magmaDoubleComplex *dwork, magma_int_t ldwork, 
               magmaDoubleComplex *hwork, magma_int_t lwork,
               double *rwork, magma_int_t lrwork,
               int *iwork, magma_int_t liwork,
               magma_solver_parameters *solver_par,
               magma_int_t *info )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======
    Solves an eigenvalue problem

       A * X = evalues X

    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory.

    This is a GPU implementation of the LOBPCG method.

    Arguments
    =========

    =====================================================================  */

#define residualNorms(i,iter)  (residualNorms + (i) + (iter)*n )
#define magma_z_bspmv(m, n, alpha, A, X, beta, AX)       {              \
        for(int k = 0; k<n; k++) {                                      \
            magma_z_vector x, ax;                                       \
            x.memory_location = Magma_CPU;  x.num_rows = m;  x.nnz = m;  x.val =  X+(k)*(m); \
            ax.memory_location= Magma_CPU; ax.num_rows = m; ax.nnz = m; ax.val = AX+(k)*(m); \
            magma_z_spmv(alpha, A, x, beta, ax );                       \
        }                                                               \
}
#define gramA(    m, n)   (gramA     + (m) + (n)*gramDim)
#define gramB(    m, n)   (gramB     + (m) + (n)*gramDim)
#define gevectors(m, n)   (gevectors + (m) + (n)*gramDim) 
#define h_gramB(  m, n)   (h_gramB   + (m) + (n)*gramDim)

    magma_int_t verbosity = 1;

    magmaDoubleComplex *blockP, *blockAP, *blockR, *blockAR, *blockAX, *blockW, *pointer;
    magmaDoubleComplex *gramA, *gramB, *gramM;
    magmaDoubleComplex *gevectors, * h_gramB;

    // Set some defaults
    double residualTolerance  = 1e-6;
    magma_int_t maxIterations = 5000;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;

    double *residualNorms, *condestGhistory, condestG;
    int* activeMask = new int[n];
    magma_dmalloc_cpu(&residualNorms, (maxIterations+1) * n);
    magma_dmalloc_cpu(&condestGhistory, maxIterations+1);

    for(int i =0; i<n; i++)
        activeMask[i] = 1;

    // Rearange the work space
    blockAX = dwork;
    blockAR = blockAX +   m * n;
    blockAP = blockAR +   m * n;
    blockR  = blockAP +   m * n;
    blockP  = blockR  +   m * n;
    blockW  = blockP  +   m * n;
    dwork   = blockW  +   m * n;

    *info = 0;
    if (m < 2)
        *info = -1;
    else if (n > m)
        *info = -2; 

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // Make the initial vectors orthonormal
    magma_zgegqr_gpu( m, n, blockX, m, dwork, hwork, info );
      
    magma_z_bspmv(m, n, c_one, A, blockX, c_zero, blockAX );
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  blockX, m, blockAX, m,
                c_zero, gramM, n);

    magma_zheevd_gpu( MagmaVec, MagmaUpper,
                      n, gramM, n, evalues,
                      dwork, n, hwork, lwork,
                      #if defined(PRECISION_z) || defined(PRECISION_c)
                      rwork, lrwork,
                      #endif
                      iwork, liwork, info );
   
    // Update  X =  X * evectors
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one,  blockX, m, gramM, n,
                c_zero, blockW, m);
    pointer = blockX;
    blockX  = blockW;
    blockW  = pointer;

    // Update AX = AX * evectors
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one,  blockAX, m, gramM, n,
                c_zero, blockW, m);
    pointer = blockAX;
    blockX  = blockW;
    blockW  = pointer;

    condestGhistory[1] = 7.82;
    magma_int_t iterationNumber, cBlockSize = n, restart    = 1, b1;
    magma_int_t iter, activePSize, gramDim;

    // Main LOBPCG loop
    for(iterationNumber = 1; iterationNumber < maxIterations; iterationNumber++)
        {
            // Compute the residuals (R = Ax - x evalues )
            magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, blockR, m);
            for(int i=0; i<n; i++){
                magma_zaxpy(m, MAGMA_Z_MAKE(-evalues[i],0), blockX+i*m, 1, blockR+i*m, 1);
            }
            magmablas_dznrm2_cols(m, n, blockR, m, residualNorms + n*iterationNumber);
            magma_zcompact(m, n, blockR, m,
                           residualNorms + n*iterationNumber, residualTolerance, 
                           activeMask, &cBlockSize);

            if (cBlockSize == 0)
                return *info;
        
            /* Apply the preconditioner P to the active residulas             */
            // magma_z_bspmv(m, cBlockSize, c_one, P, blockR, c_zero, blockW ); 
            // For now set P to be identity (no preconditioner)
            magmablas_zlacpy( MagmaUpperLower, m, cBlockSize, blockR, m, blockW, m);

            pointer = blockR;
            blockR  = blockW;
            blockW  = pointer;
            
            // Make the active preconditioned residuals R orth. to X
            // Do I need this???

            // Make the active preconditioned residuals orthonormal
            magma_zgegqr_gpu( m, cBlockSize, blockR, m, dwork, hwork, info );

            magma_z_bspmv(m, cBlockSize, c_one, A, blockR, c_zero, blockAR );

            magma_zcompactActive(m, n, blockP,  m, activeMask);
            magma_zcompactActive(m, n, blockAP, m, activeMask);
         
            if (iterationNumber>1){
                // Make the active P orthonormal & properly change AP (without multiplication by A)
                magma_zgegqr_gpu( m, cBlockSize, blockP, m, dwork, hwork, info );
                magma_zsetmatrix( cBlockSize, cBlockSize, hwork, cBlockSize, dwork, cBlockSize);
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                             m, cBlockSize, c_one, dwork, cBlockSize, blockAP, m);
            }

            iter = max(1,iterationNumber-10- (int)(log(1.*cBlockSize)));
            double condestGmean = 0.;
            for(int i = 0; i<iterationNumber-iter+1; i++)
                condestGmean += condestGhistory[i];
            condestGmean = condestGmean / (iterationNumber-iter+1);
         
            if (iterationNumber == 1){
                activePSize = 0;
                restart     = 1;
            }
            else{
                activePSize = cBlockSize;
                restart     = 0;
            }

            gramDim = n+2*cBlockSize;

            /* --- The Raileight-Ritz method for [X R P] -----------------------
               [ X R P ]'  [AX  AR  AP] y = evalues [ X R P ]' [ X R P ], i.e.,

                / X'AX  X'AR  X'AP \                 / X'X  X'R  X'P \
               |  R'AX  R'AR  R'AP  | y   = evalues |  R'X  R'R  R'P  |
                \ P'AX  P'AR  P'AP /                 \ P'X  P'R  P'P /       
               -----------------------------------------------------------------   */
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                        c_one, blockR, m, blockAX, m, c_zero, gramA(n,0), gramDim);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m,
                        c_one, blockR, m, blockAR, m, c_zero, gramA(n,n), gramDim);

            memset( gevectors(0,0), 0, n*n*sizeof(magmaDoubleComplex) );
            for(int k=0; k<n; k++)
                *gevectors(k,k) = MAGMA_Z_MAKE(evalues[k], 0);

            memset( h_gramB(0,0), 0, gramDim*gramDim*sizeof(magmaDoubleComplex) );
            for(int k=0; k<gramDim; k++)
                *h_gramB(k,k) = MAGMA_Z_MAKE(1., 0.);
            
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                        c_one, blockP, m, blockX, m, c_zero, gramB(n+cBlockSize,0), gramDim);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m,
                        c_one, blockR, m, blockX, m, c_zero, gramB(n,0), gramDim);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, cBlockSize, m,
                        c_one, blockP, m, blockR, m, c_zero, gramB(n+cBlockSize,n), gramDim);

            // Get GramB from the GPU to the CPU and compute its eigenvalues only
            magma_zgetmatrix(gramDim, gramDim, gramB, gramDim, h_gramB, gramDim);
            lapackf77_zheev("N", "U", &gramDim, h_gramB, &gramDim, evalues,
                            hwork, &lwork,
                            #if defined(PRECISION_z) || defined(PRECISION_c)
                            rwork, 
                            #endif
                            info);

            condestG = log10( evalues[gramDim-1]/evalues[0] ) + 1.;
           
            if ((condestG/condestGmean>2 && condestG>2) || condestG>8) {
                // Steepest descent restart for stability
                restart=1;
                printf("restart at step #%d\n", iterationNumber);
            }

            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m, 
                        c_one, blockP, m, blockAX, m, c_zero, 
                        gramA(n+cBlockSize,0), gramDim);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m, 
                        c_one, blockP, m, blockAR, m, c_zero, 
                        gramA(n+cBlockSize,n), gramDim);
            magma_zgemm(MagmaConjTrans, MagmaNoTrans, cBlockSize, n, m, 
                        c_one, blockP, m, blockAP, m, c_zero, 
                        gramA(n+cBlockSize,n+cBlockSize), gramDim);
            
            if (restart==0) {
                magma_zgetmatrix(gramDim, gramDim, gramA, gramDim, gevectors, gramDim);
            }
            else {
                gramDim = n+cBlockSize;
                magma_zgetmatrix(gramDim, gramDim, gramA, gramDim, gevectors, gramDim);
            }
            magma_int_t itype = 1;
            lapackf77_zhegvd(&itype, "V", "U", &gramDim,
                             gevectors, &gramDim, gramB, &gramDim,
                             evalues, hwork, &lwork, 
                             #if defined(PRECISION_z) || defined(PRECISION_c)
                             rwork, &lrwork,
                             #endif
                             iwork, &liwork, info);

            if (restart == 0) {
                b1 = n+cBlockSize;
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockP, m, 
                            gevectors + b1, gramDim, c_zero, dwork, m);
                blockP = dwork;
 
                b1 = n;
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockR, m,
                            gevectors + b1, gramDim, c_one, blockP, m);

                b1 = n+cBlockSize;
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockAP, m,
                            gevectors + b1, gramDim, c_zero, dwork, m);
                blockAP = dwork;

                b1 = n;
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockAR, m, 
                            gevectors + b1, gramDim, c_one, blockAP, m);
            }
            else {
                magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, cBlockSize,
                            c_one, blockR, m,
                            gevectors + n, gramDim, c_zero, blockP, m);
                magma_zgemm(MagmaNoTrans, MagmaNoTrans,m, n, cBlockSize,
                            c_one, blockAR, m,
                            gevectors +n, gramDim, c_zero, blockAP, m);
            }

            magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                        c_one, blockX, m, 
                        gevectors, gramDim, c_zero, dwork, m);
            // blockX  = dwork + blockP;
            magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockX, m);
            magma_zaxpy(m*n, c_one, blockP, 1, blockX, 1);

            magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                        c_one, blockAX, m,
                        gevectors, gramDim, c_zero, dwork, m);
            // blockAX  = dwork + blockAP;
            magmablas_zlacpy( MagmaUpperLower, m, n, dwork, m, blockAX, m);
            magma_zaxpy(m*n, c_one, blockAP, 1, blockAX, 1);

            condestGhistory[iterationNumber+1]=condestG;
            if (verbosity==1) {
                printf("Iteration %d, CBS %d, Residual %f\n",
                       iterationNumber, cBlockSize, *residualNorms(1,iterationNumber));
            }

        }   //  end for iterationNumber = 1,maxIterations ------------
    
    // -----------------------------------------------------------------------------
    // --- postprocessing;
    // -----------------------------------------------------------------------------

    magma_z_bspmv(m, n, c_one, A, blockX, c_zero, blockAX );
    magma_zgemm(MagmaConjTrans, MagmaNoTrans, n, n, m,
                c_one,  blockX, m, blockAX, m,
                c_zero, gramM, n);

    magma_zheevd_gpu( MagmaVec, MagmaUpper,
                      n, gramM, n, evalues,
                      dwork, n, hwork, lwork, 
                      #if defined(PRECISION_z) || defined(PRECISION_c)
                      rwork, lrwork,
                      #endif
                      iwork, liwork, info );
    // call eig(evalues, evectors, gramM, n);

    // Update X = X * evectors
    magmablas_zlacpy( MagmaUpperLower, m, n, blockX, m, dwork, m);
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one, dwork, m,
                gramM, n, c_zero, blockX, m);

    // Update AX = AX * evectors
    magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, dwork, m);
    magma_zgemm(MagmaNoTrans, MagmaNoTrans, m, n, n,
                c_one, dwork, m,
                gramM, n, c_zero, blockAX, m);

    // Compute R = AX - evalues X
    magmablas_zlacpy( MagmaUpperLower, m, n, blockAX, m, blockR, m);
    for(int i=0; i<n; i++)
        magma_zaxpy(m, MAGMA_Z_MAKE(-evalues[i], 0), blockX, 1, blockR, 1);

    // residualNorms[iterationNumber] = || R ||    
    magmablas_dznrm2_cols(m, n, blockR, m, residualNorms+ n*iterationNumber);

    printf("Eigenvalues:\n");
    for(int i =0; i<n; i++)
        printf("%f  ", evalues[i]);
    printf("\n\n");

    printf("Final residuals:\n");
    for(int i =0; i<n; i++)
        printf("%f  ", *residualNorms(i,iterationNumber));
    printf("\n\n");

    printf("Residuals are stored in file residualNorms\n");
    printf("Plot the residuals using: myplot \n");

    FILE *residuals_file;
    residuals_file = fopen("residualNorms", "w");
    for(int i =0; i<iterationNumber; i++) {
        for(int j = 0; j<n; j++)
            fprintf(residuals_file, "%f \n", *residualNorms(j,i));
        fprintf(residuals_file, "\n");
    }
    fclose(residuals_file);
 
    magma_free_cpu( residualNorms );
    magma_free_cpu( condestGhistory );
    delete activeMask;

    return MAGMA_SUCCESS;
}
