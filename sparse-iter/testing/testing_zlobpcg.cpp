/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> c d s
       @author Stan Tomov
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "../include/magmasparse.h"
#include "testings.h"

extern "C" magma_int_t
magma_zlobpcg( magma_int_t m, magma_int_t n, magma_z_sparse_matrix A,
               magmaDoubleComplex *blockX, double *evalues,
               magmaDoubleComplex *dwork, magma_int_t ldwork,
               magmaDoubleComplex *hwork, magma_int_t lwork,
               magma_solver_parameters *solver_par, magma_int_t *info );


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing magma_zlobpcg
*/
int main( int argc, char** argv)
{
    TESTING_INIT();

    real_Double_t  gpu_time;
    magma_opts opts;
    parse_opts( argc, argv, &opts );
    magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;

    magma_z_sparse_matrix A;
    magma_z_sparse_matrix dA;

    magma_int_t m         = opts.msize[0];
    magma_int_t blockSize = 10; 

    // Initialize a matrix to be 2 on the diagonal and -1 on subdiagonals
    magma_int_t offdiags = 1;
    magma_int_t *diag_offset;
    magmaDoubleComplex *diag_vals;
    magma_zmalloc_cpu( &diag_vals, offdiags+1 );
    magma_imalloc_cpu( &diag_offset, offdiags+1 );

    diag_offset[0] = 0;
    diag_offset[1] = 1;

    diag_vals[0] = MAGMA_Z_MAKE( 2.0, 0.0 );
    diag_vals[1] = MAGMA_Z_MAKE( -1.0, 0.0 );

    magma_zmgenerator( m, offdiags, diag_offset, diag_vals, &A );
    magma_z_mtransfer(A, &dA, Magma_CPU, Magma_DEV);

    // Memory allocation for the eigenvectors, eigenvalues, and workspace
    double *evalues;
    magmaDoubleComplex *evectors, *hevectors, *dwork, *hwork;
    magma_int_t info, ldwork = 8*m*blockSize;
    magma_int_t lhwork = max(2*blockSize+blockSize*magma_get_dsytrd_nb(blockSize),
                             1 + 6*3*blockSize + 2* 3*blockSize* 3*blockSize);

    // This to be revisited - return is just blockSize but we use this for the
    // generalized eigensolver as well so we need 3X the memory
    magma_dmalloc_cpu(    &evalues ,     3*blockSize );

    magma_zmalloc(        &evectors, m * blockSize );
    magma_zmalloc_cpu(   &hevectors, m * blockSize );
    magma_zmalloc(        &dwork   ,        ldwork );
    magma_zmalloc_pinned( &hwork   ,        lhwork );

    // Solver parameters
    magma_solver_parameters solver_par;
    solver_par.epsilon = 1e-3;
    solver_par.maxiter = 360;
    
    magma_int_t n2 = m * blockSize;
    lapackf77_zlarnv( &ione, ISEED, &n2, hevectors );
    magma_zsetmatrix( m, blockSize, hevectors, m, evectors, m );

    // Find the blockSize smallest eigenvalues and corresponding eigen-vectors
    gpu_time = magma_wtime();
    magma_zlobpcg( m, blockSize, 
                   dA, evectors, evalues,
                   dwork, ldwork,
                   hwork, lhwork,
                   &solver_par, &info);
    gpu_time = magma_wtime() - gpu_time;

    printf("Time (sec) = %7.2f\n", gpu_time);

    magma_z_mfree(     &A    );
    magma_free_cpu(  evalues );
    magma_free(     evectors );
    magma_free_cpu( hevectors);
    magma_free(     dwork    );
    magma_free_pinned( hwork    );

    TESTING_FINALIZE();
    return 0;
}
