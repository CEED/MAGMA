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
    A           magma_z_matrix
                sparse matrix A    

    @param[in]
    b           magma_z_matrix
                input vector b     

    @param[in]
    x           magma_z_matrix*
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
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_zopts *zopts,
    magma_queue_t queue )
{
    // make sure RHS is a dense matrix
    if ( b.storage_type != Magma_DENSE ) {
        magma_z_matrix bdense;
        magma_zmconvert( b, &bdense, b.storage_type, Magma_DENSE, queue );
        magma_zmfree(&b, queue);
        magma_zmtranspose(bdense, &b, queue );
        magma_zmfree(&bdense, queue);    
    }
    if( b.num_cols == 1 ){
    // preconditioner
        if ( zopts->solver_par.solver != Magma_ITERREF ) {
            int stat = magma_z_precondsetup( A, b, &zopts->precond_par, queue );
            if (  stat != MAGMA_SUCCESS ){ 
                printf("error: bad preconditioner.\n");
                return MAGMA_ERR_BADPRECOND;
            }
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
                    magma_zfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_PGMRES: 
                    magma_zfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_LOBPCG: 
                    magma_zlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_ITERREF:
                    magma_ziterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_JACOBI: 
                    magma_zjacobi( A, b, x, &zopts->solver_par, queue );break;
            case  Magma_BAITER: 
                    magma_zjacobidomainoverlap( A, b, x, &zopts->solver_par, queue );break;
                    //magma_zbaiter( A, b, x, &zopts->solver_par, queue );break;
            default:  
                    printf("error: solver class not supported.\n");break;
        }
    }
    else{
  // preconditioner
        if ( zopts->solver_par.solver != Magma_ITERREF ) {
            int stat = magma_z_precondsetup( A, b, &zopts->precond_par, queue );
            if (  stat != MAGMA_SUCCESS ){ 
                printf("error: bad preconditioner.\n");
                return MAGMA_ERR_BADPRECOND;
            }
        }
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    magma_zbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_PCG:
                    magma_zbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue );break;
            case  Magma_LOBPCG: 
                    magma_zlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue );break;
            default:  
                    printf("error: only 1 RHS supported for this solver class.\n");break;
        }   
    }
    return MAGMA_SUCCESS;
}


