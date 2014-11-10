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
#include "magmasparse.h"




/**
    Purpose
    -------

    ALlows the user to choose a solver.

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_vector
                input vector b     

    @param[in]
    x           magma_z_vector*
                output vector x        

    @param[in]
    zopts     magma_zopts
              options for solver and preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_solver(
    magma_z_sparse_matrix A, magma_z_vector b, 
    magma_z_vector *x, magma_zopts *zopts,
    magma_queue_t queue )
{
    // preconditioner
        if ( zopts->solver_par.solver != Magma_ITERREF ) {
            magma_z_precondsetup( A, b, &zopts->precond_par, queue );
        }
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    magma_zcg_res( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_CGMERGE:
                    magma_zcg_merge( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PCG:
                    magma_zpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_BICGSTAB:
                    magma_zbicgstab( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_BICGSTABMERGE: 
                    magma_zbicgstab_merge( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PBICGSTAB: 
                    magma_zpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_GMRES: 
                    magma_zgmres( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_PGMRES: 
                    magma_zpgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_LOBPCG: 
                    magma_zlobpcg( A, &zopts->solver_par, queue );break;
            case  Magma_ITERREF:
                    magma_ziterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_JACOBI: 
                    magma_zjacobi( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_BAITER: 
                    magma_zbaiter( A, b, x, &zopts->solver_par, queue );break;
            
        }
    return MAGMA_SUCCESS;
}


