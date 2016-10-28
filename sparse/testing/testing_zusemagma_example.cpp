/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal d -> s
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- This is an example how magma can be integrated into another software.
*/
int main ()
{
    // The software does e.g. discretization of a PDE,
    // ends up with a sparse linear system in CSR format and a RHS.
    // Let's assume this system is a diagonal system of size m.
    
    int m=200, n=1;
    int *col, *row;
    double *val, *rhs, *sol;
    
    row = (int*) calloc (m+1, sizeof(int));
    col = (int*) calloc (m, sizeof(int));
    
    val = (double*) calloc (m, sizeof(double));
    rhs = (double*) calloc (m, sizeof(double));
    sol = (double*) calloc (m, sizeof(double));
    
    for( int i=0; i<m; i++ ){
        col[i] = i;
        row[i] = i;
        val[i] = 55.0;
        rhs[i] = 3.0;
        sol[i] = 0.0;
    }
    row[m] = m;
        
    // We now initialize MAGMA and create some LA structures.
    magma_init();
    magma_dopts opts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    
    magma_d_matrix A={Magma_CSR}, dA={Magma_CSR};
    magma_d_matrix b={Magma_CSR}, db={Magma_CSR};
    magma_d_matrix x={Magma_CSR}, dx={Magma_CSR}; 
    
    // We now pass the system to MAGMA.
    magma_dcsrset( m, m, row, col, val, &A, queue );
    magma_dvset( m, 1, rhs, &b, queue );
    magma_dvset( m, 1, sol, &x, queue );
    
    // We now choose a solver, preconditioner etc - see documentation for options.
    opts.solver_par.solver = Magma_PIDRMERGE;
    opts.solver_par.restart = 8;
    opts.solver_par.maxiter = 1000;
    opts.solver_par.rtol = 1e-10;
    opts.solver_par.maxiter = 1000;
    opts.precond_par.solver = Magma_ILU;
    opts.precond_par.levels = 0;
    opts.precond_par.trisolver = Magma_CUSOLVE;
    
    // We initialize the solver.
    magma_dsolverinfo_init( &opts.solver_par, &opts.precond_par, queue );
    
    // We copy the system to the device (optional, only necessary if using the GPU)
    magma_dmtransfer( A, &dA, Magma_CPU, Magma_DEV, queue );
    magma_dmtransfer( b, &db, Magma_CPU, Magma_DEV, queue );
    magma_dmtransfer( x, &dx, Magma_CPU, Magma_DEV, queue );
    
    // We now generate the preconditioner.
    magma_d_precondsetup( dA, db, &opts.solver_par, &opts.precond_par, queue );
    
    // In case we only wanted to generate a preconditioner, we are done.
    // The preconditioner in the opts.precond_par structure - in this case an ILU.
    // The lower ILU(0) factor is in opts.precond_par.L and
    // the upper ILU(0) factor is in opts.precond_par.U (in this case on the device).
    
    // If we want to solve the problem, we run:
    magma_d_solver( dA, db, &dx, &opts, queue );

    // Then we copy the solution back to the host...
    magma_dmfree( &x, queue );
    magma_dmtransfer( dx, &x, Magma_CPU, Magma_DEV, queue );
    
    // and back to the application code
    magma_dvget( x, &m, &n, &sol, queue );
    
    // We free the allocated memory...
    magma_dmfree( &dx, queue );
    magma_dmfree( &db, queue );
    magma_dmfree( &dA, queue );
    
    // and finalize MAGMA.
    magma_queue_destroy( queue );
    magma_finalize();
    
    // From here on, the application code may continue with the solution in sol...
    
    for(int i=0; i<20; i++){
        printf("%.4f\n", sol[i]);   
    }
}
