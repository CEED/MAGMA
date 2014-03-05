/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../../include/magmablas.h"
#include "../include/magmasparse_types.h"
#include "../include/magmasparse.h"




/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

    Arguments
    =========

    magma_z_sparse_matrix A         sparse matrix A    
    magma_z_vector x                input vector x  
    magma_z_vector y                input vector y      
    magma_z_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_z_precond( magma_z_sparse_matrix A, magma_z_vector b, 
                 magma_z_vector *x, magma_z_preconditioner precond )
{
// set up precond parameters as solver parameters   
    magma_z_solver_par psolver_par;
    psolver_par.epsilon = precond.epsilon;
    psolver_par.maxiter = precond.maxiter;
    psolver_par.restart = precond.restart;
   
    if( precond.solver == Magma_CG ){
// printf( "start CG preconditioner with epsilon: %f and maxiter: %d: ", 
//                            psolver_par.epsilon, psolver_par.maxiter );
        magma_zcg( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_GMRES ){
// printf( "start GMRES preconditioner with epsilon: %f and maxiter: %d: ", 
//                               psolver_par.epsilon, psolver_par.maxiter );
        magma_zgmres( A, b, x, &psolver_par );
// printf( "done.\n" );
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BICGSTAB ){
// printf( "start BICGSTAB preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zbicgstab( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_JACOBI ){
// printf( "start JACOBI preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zjacobi( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }
    if( precond.solver == Magma_BCSRLU ){
// printf( "start BCSRLU preconditioner with epsilon: %f and maxiter: %d: ", 
//                                  psolver_par.epsilon, psolver_par.maxiter );
        magma_zbcsrlu( A, b, x, &psolver_par );
// printf( "done.\n");
        return MAGMA_SUCCESS;
    }

    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is preprocessed. 
    E.g. for Jacobi: the scaling-vetor, for ILU the factorization...

    Arguments
    =========

    magma_z_sparse_matrix A         sparse matrix A    
    magma_z_vector x                input vector x  
    magma_z_vector y                input vector y      
    magma_z_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_z_precondsetup( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_zjacobisetup_diagscal( A, &(precond->d) );
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_PASTIX ){
        magma_zpastixsetup( A, b, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}



/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

    Purpose
    =======

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is applied. 
    E.g. for Jacobi: the scaling-vetor, for ILU the triangular solves...

    Arguments
    =========

    magma_z_sparse_matrix A         sparse matrix A    
    magma_z_vector x                input vector x  
    magma_z_vector y                input vector y      
    magma_z_preconditioner precond  preconditioner

    ========================================================================  */

magma_int_t
magma_z_applyprecond( magma_z_sparse_matrix A, magma_z_vector b, 
                      magma_z_vector *x, magma_z_preconditioner *precond )
{
    if( precond->solver == Magma_JACOBI ){
        magma_zjacobi_diagscal( A.num_rows, precond->d.val, b.val, x->val );
        return MAGMA_SUCCESS;
    }
    if( precond->solver == Magma_PASTIX ){
        magma_zapplypastix( b, x, precond );
        return MAGMA_SUCCESS;
    }
    else{
        printf( "error: preconditioner type not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }

}
