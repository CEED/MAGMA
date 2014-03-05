/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Prints information about a previously called solver.

    Arguments
    =========

    magma_z_solver_par *solver_par    structure containing all information

    ========================================================================  */


magma_int_t
magma_zsolverinfo( magma_z_solver_par *solver_par, 
                    magma_z_preconditioner *precond_par ){

    if( (solver_par->solver == Magma_CG) || (solver_par->solver == Magma_PCG) ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            if( solver_par->solver == Magma_CG )
                printf("#   CG performance analysis every %d iteration\n", k);
            else if( solver_par->solver == Magma_PCG ){
                if( precond_par->solver == Magma_JACOBI )
                        printf("#   Jacobi-CG performance analysis"
                                " every %d iteration\n", k);

            }
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# CG solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_CGMERGE ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   CG (merged) performance analysis every %d iteration\n",
                                                                             k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# CG (merged) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTAB ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   BiCGStab performance analysis every %d iteration\n",
                                                                             k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTABMERGE ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   BiCGStab (merged) performance analysis"
                   " every %d iteration\n", k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab (merged) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BICGSTABMERGE2 ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("#   BiCGStab (merged2) performance analysis"
                   " every %d iteration\n", k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# BiCGStab (merged2) solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_GMRES ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("# GMRES-(%d) performance analysis"
                   " every %d iteration\n",solver_par->restart, k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# GMRES-(%d) solver summary:\n", solver_par->restart);
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_ITERREF ){
        if( solver_par->verbose > 0 ){
            magma_int_t k = solver_par->verbose;
            printf("#======================================================="
                    "======#\n");
            printf("# Iterative Refinement performance analysis"
                   " every %d iteration\n", k);
            printf("#   iter   ||   residual-nrm2    ||   runtime \n");
            printf("#======================================================="
                    "======#\n");
            for( int j=0; j<(solver_par->numiter)/k+1; j++ ){
                printf("   %4d    ||    %e    ||    %f\n", 
                            j*k, solver_par->res_vec[j], solver_par->timing[j]);
            }
        }
        printf("#======================================================="
                "======#\n");
        printf("# Iterative Refinement solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_JACOBI ){
        printf("#======================================================="
                "======#\n");
        printf("# Jacobi solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    iterations: %4d\n", (solver_par->numiter) );
        printf("#    exact final residual: %e\n#    runtime: %.4f sec\n", 
                    solver_par->final_res, solver_par->runtime);
        printf("#======================================================="
                "======#\n");
    }else if( solver_par->solver == Magma_BCSRLU ){
        printf("#======================================================="
                "======#\n");
        printf("# BCSRLU solver summary:\n");
        printf("#    initial residual: %e\n", solver_par->init_res );
        printf("#    exact final residual: %e\n", solver_par->final_res );
        printf("#    runtime factorization: %4lf sec\n",
                    solver_par->timing[0] );
        printf("#    runtime triangular solve: %.4f sec\n", 
                    solver_par->timing[1] );
        printf("#======================================================="
                "======#\n");
    }else{
        printf("error: solver info not supported.\n");
    }

    return MAGMA_SUCCESS;
}


/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Frees any memory assocoiated with the verbose mode of solver_par. The
    other values are set to default.

    Arguments
    =========

    magma_z_solver_par *solver_par    structure containing all information

    ========================================================================  */


magma_int_t
magma_zsolverinfo_free( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond ){

    if( solver_par->res_vec != NULL ){
        magma_free_cpu( solver_par->res_vec );
        solver_par->res_vec = NULL;
    }
    if( solver_par->timing != NULL ){
        magma_free_cpu( solver_par->timing );
        solver_par->timing = NULL;
    }
    if( solver_par->eigenvectors != NULL ){
        magma_free( solver_par->eigenvectors );
        solver_par->eigenvectors = NULL;
    }
    if( solver_par->eigenvalues != NULL ){
        magma_free_cpu( solver_par->eigenvalues );
        solver_par->eigenvalues = NULL;
    }

    if( precond->d.val != NULL ){
        magma_free( precond->d.val );
        precond->d.val = NULL;
    }
    if( precond->M.val != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.val );
        else
            magma_free_cpu( precond->M.val );
        precond->M.val = NULL;
    }
    if( precond->M.col != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.col );
        else
            magma_free_cpu( precond->M.col );
        precond->M.col = NULL;
    }
    if( precond->M.row != NULL ){
        if ( precond->M.memory_location == Magma_DEV )
            magma_free( precond->M.row );
        else
            magma_free_cpu( precond->M.row );
        precond->M.row = NULL;
    }
    if( precond->M.blockinfo != NULL ){
        magma_free_cpu( precond->M.blockinfo );
        precond->M.blockinfo = NULL;
    }

    return MAGMA_SUCCESS;
}


magma_int_t
magma_zsolverinfo_init( magma_z_solver_par *solver_par, 
                        magma_z_preconditioner *precond ){

/*
    solver_par->solver = Magma_CG;
    solver_par->maxiter = 1000;
    solver_par->numiter = 0;
    solver_par->ortho = Magma_CGS;
    solver_par->epsilon = RTOLERANCE;
    solver_par->restart = 30;
    solver_par->init_res = 0.;
    solver_par->final_res = 0.;
    solver_par->runtime = 0.;
    solver_par->verbose = 0;
    solver_par->info = 0;
*/
    magma_int_t iterblock = solver_par->verbose;
    if( solver_par->verbose > 0 ){
        magma_malloc_cpu( (void **)&solver_par->res_vec, sizeof(real_Double_t) 
                * ( (solver_par->maxiter)/(solver_par->verbose)+1) );
        magma_malloc_cpu( (void **)&solver_par->timing, sizeof(real_Double_t) 
                *( (solver_par->maxiter)/(solver_par->verbose)+1) );
    }else{
        solver_par->res_vec = NULL;
        solver_par->timing = NULL;
    }  

    if( solver_par->num_eigenvalues > 0 ){
        magma_dmalloc_cpu( &solver_par->eigenvalues , 
                                3*solver_par->num_eigenvalues );

        // setup initial guess EV using lapack
        // then copy to GPU
        magma_int_t ev = solver_par->num_eigenvalues * solver_par->ev_length;
        magmaDoubleComplex *initial_guess;
        magma_zmalloc_cpu( &initial_guess, ev );
        magma_zmalloc( &solver_par->eigenvectors, ev );
        magma_int_t ISEED[4] = {0,0,0,1}, ione = 1;
        lapackf77_zlarnv( &ione, ISEED, &ev, initial_guess );
        magma_zsetmatrix( solver_par->ev_length, solver_par->num_eigenvalues, 
            initial_guess, solver_par->ev_length, solver_par->eigenvectors, 
                                                    solver_par->ev_length );

        magma_free_cpu( initial_guess );
    }else{
        solver_par->eigenvectors = NULL;
        solver_par->eigenvalues = NULL;
    }  

    precond->d.val = NULL;
    precond->M.val = NULL;
    precond->M.col = NULL;
    precond->M.row = NULL;
    precond->M.blockinfo = NULL;

    return MAGMA_SUCCESS;
}


