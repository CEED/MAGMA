/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @author Stan Tomov
       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"

#include <cblas.h>

#define PRECISION_z

#define  q(i)     (q.val + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*(1+ldh)]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) dH[(i)   + (j)*(1+ldh)]


#define RTOLERANCE     1e-16
#define ATOLERANCE     1e-16

// uncomment for chronometry
#define ENABLE_TIMER
#define iterblock 1


/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the GMRES method.

    Arguments
    =========

    magma_z_sparse_matrix A                   descriptor for matrix A
    magma_z_vector b                          RHS b vector
    magma_z_vector *x                         solution approximation
    magma_solver_parameters *solver_par       solver parameters

    ========================================================================  */

magma_int_t
magma_zgmres( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
              magma_solver_parameters *solver_par ){

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, 
                                                c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m = 0, iter, restart = min( dofs-1, solver_par->restart );
    magma_int_t ldh = restart+1;
    double nom, rNorm, RNorm, den, nom0, betanom, r0 = 0.;

    // CPU workspace
    magma_setdevice(0);
    magmaDoubleComplex *H, *HH, *y, *h1;
    magma_zmalloc_pinned( &H, (ldh+1)*ldh );
    magma_zmalloc_pinned( &y, ldh );
    magma_zmalloc_pinned( &HH, ldh*ldh );
    magma_zmalloc_pinned( &h1, ldh );

    // GPU workspace
    magma_z_vector r, q, q_t;
    magma_z_vinit( &r, Magma_DEV, dofs, c_zero );
    magma_z_vinit( &q, Magma_DEV, dofs*(ldh+1), c_zero );
    q_t.memory_location = Magma_DEV; 
    q_t.val = NULL; 
    q_t.num_rows = q_t.nnz = dofs;

    magmaDoubleComplex *dy, *dH = NULL;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    if (MAGMA_SUCCESS != magma_zmalloc( &dH, (ldh+1)*ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;

    // GPU stream
    magma_queue_t stream[2];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_event_create( &event[0] );

  
    magmablasSetKernelStream(stream[0]);
    magma_zscal( dofs, c_zero, x->val, 1 );              //  x = 0
    magma_zcopy( dofs, b.val, 1, r.val, 1 );             //  r = b
    r0 = magma_dznrm2( dofs, r.val, 1 );                 //  r0= || r||
    nom = nom0 = r0*r0;
    H(1,0) = MAGMA_Z_MAKE( r0, 0. ); 
    magma_zsetvector(1, &H(1,0), 1, &dH(1,0), 1);
    if ( (r0 = nom * solver_par->epsilon) < ATOLERANCE ) 
        r0 = ATOLERANCE;
    if ( nom < r0 )
        return MAGMA_SUCCESS;
    //Chronometry
    #ifdef ENABLE_TIMER
    double tempo1, tempo2;
    magma_device_sync(); tempo1=magma_wtime();
    printf("#=============================================================#\n");
    printf("#   GMRES-(%d) performance analysis every %d iteration   \n", restart, iterblock);
    printf("#   iter   ||   residual-nrm2    ||   runtime    \n");
    printf("#=============================================================#\n");
    printf("      0    ||    %e    ||    0.0000      \n", nom);
    magma_device_sync(); tempo1=magma_wtime();
    #endif
    
    // start iteration
    for( solver_par->numiter= 1; solver_par->numiter<solver_par->maxiter; 
                                                    solver_par->numiter++ ){
        magma_zcopy(dofs, r.val, 1, q(0), 1);       //  q[0]    = 1.0/||r||
        magma_zscal(dofs, 1./H(1,0), q(0), 1);      //  (to be fused)

        for(k=1; k<=restart; k++) {
            q_t.val = q(k-1);
            magmablasSetKernelStream(stream[0]);
            magma_z_spmv( c_one, A, q_t, c_zero, r );
                 // r = A q[k] 
            if (solver_par->ortho == Magma_MGS ) {
                // modified Gram-Schmidt
                magmablasSetKernelStream(stream[0]);
                for (i=1; i<=k; i++) {
                    H(i,k) =magma_zdotc(dofs, q(i-1), 1, r.val, 1);            
                        //  H(i,k) = q[i] . r
                    magma_zaxpy(dofs,-H(i,k), q(i-1), 1, r.val, 1);            
                       //  r = r - H(i,k) q[i]
                }
                H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );
                      //  H(k+1,k) = sqrt(r . r) 
                if (k < restart) {
                        magma_zcopy(dofs, r.val, 1, q(k), 1);                  
                      //  q[k] = 1.0/H[k][k-1] r
                        magma_zscal(dofs, 1./H(k+1,k), q(k), 1);               
                      //  (to be fused)   
                 }
            } else if (solver_par->ortho == Magma_FUSED_CGS ) {
                // fusing zgemv with dznrm2 in classical Gram-Schmidt
                magmablasSetKernelStream(stream[0]);
                magma_zcopy(dofs, r.val, 1, q(k), 1);  
                    // dH(1:k+1,k) = q[0:k] . r
                magmablas_zgemv(MagmaTrans, dofs, k+1, c_one, q(0), 
                                dofs, r.val, 1, c_zero, &dH(1,k), 1);
                    // r = r - q[0:k-1] dH(1:k,k)
                magmablas_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), 
                                dofs, &dH(1,k), 1, c_one, r.val, 1);
                   // 1) dH(k+1,k) = sqrt( dH(k+1,k) - dH(1:k,k) )
                magma_zcopyscale(  dofs, k, r.val, q(k), &dH(1,k) );  
                   // 2) q[k] = q[k] / dH(k+1,k) 

                magma_event_record( event[0], stream[0] );
                magma_queue_wait_event( stream[1], event[0] );
                magma_zgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]); 
                    // asynch copy dH(1:(k+1),k) to H(1:(k+1),k)
            } else {
                // classical Gram-Schmidt (default)
                // > explicitly calling magmabls
                magmablasSetKernelStream(stream[0]);                                                  
                magmablas_zgemv(MagmaTrans, dofs, k, c_one, q(0), 
                                dofs, r.val, 1, c_zero, &dH(1,k), 1); 
                                // dH(1:k,k) = q[0:k-1] . r
                #ifndef DZNRM2SCALE 
                // start copying dH(1:k,k) to H(1:k,k)
                magma_event_record( event[0], stream[0] );
                magma_queue_wait_event( stream[1], event[0] );
                magma_zgetvector_async(k, &dH(1,k), 1, &H(1,k), 
                                                    1, stream[1]);
                #endif
                                  // r = r - q[0:k-1] dH(1:k,k)
                magmablas_zgemv(MagmaNoTrans, dofs, k, c_mone, q(0), 
                                    dofs, &dH(1,k), 1, c_one, r.val, 1);
                #ifdef DZNRM2SCALE
                magma_zcopy(dofs, r.val, 1, q(k), 1);                 
                    //  q[k] = r / H(k,k-1) 
                magma_dznrm2scale(dofs, q(k), dofs, &dH(k+1,k) );     
                    //  dH(k+1,k) = sqrt(r . r) and r = r / dH(k+1,k)

                magma_event_record( event[0], stream[0] );            
                            // start sending dH(1:k,k) to H(1:k,k)
                magma_queue_wait_event( stream[1], event[0] );        
                            // can we keep H(k+1,k) on GPU and combine?
                magma_zgetvector_async(k+1, &dH(1,k), 1, &H(1,k), 1, stream[1]);
                #else
                H(k+1,k) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. );   
                            //  H(k+1,k) = sqrt(r . r) 
                if( k<solver_par->restart ){
                        magmablasSetKernelStream(stream[0]);
                        magma_zcopy(dofs, r.val, 1, q(k), 1);                  
                            //  q[k]    = 1.0/H[k][k-1] r
                        magma_zscal(dofs, 1./H(k+1,k), q(k), 1);              
                            //  (to be fused)   
                 }
                #endif
            }
        }
        magma_queue_sync( stream[1] );
        for( k=1; k<=restart; k++ ){
            /*     Minimization of  || b-Ax ||  in H_k       */ 
            for (i=1; i<=k; i++) {
                #if defined(PRECISION_z) || defined(PRECISION_c)
                cblas_zdotc_sub( i+1, &H(1,k), 1, &H(1,i), 1, &HH(k,i) );
                #else
                HH(k,i) = cblas_zdotc(i+1, &H(1,k), 1, &H(1,i), 1);
                #endif
            }
            h1[k] = H(1,k)*H(1,0); 
            if (k != 1)
                for (i=1; i<k; i++) {
                    for (m=i+1; m<k; m++){
                        HH(k,m) -= HH(k,i) * HH(m,i);
                    }
                    HH(k,k) -= HH(k,i) * HH(k,i) / HH(i,i);
                    HH(k,i) = HH(k,i)/HH(i,i);
                    h1[k] -= h1[i] * HH(k,i);   
                }    
            y[k] = h1[k]/HH(k,k); 
            if (k != 1)  
                for (i=k-1; i>=1; i--) {
                    y[i] = h1[i]/HH(i,i);
                    for (j=i+1; j<=k; j++)
                        y[i] -= y[j] * HH(j,i);
                }                    
            m = k;
            rNorm = fabs(MAGMA_Z_REAL(H(k+1,k)));
        }

        magma_zsetmatrix_async(m, 1, y+1, m, dy, m, stream[0]);
        magmablasSetKernelStream(stream[0]);
        magma_zgemv(MagmaNoTrans, dofs, m, c_one, q(0), dofs, dy, 1, 
                                                    c_one, x->val, 1); 
        magma_z_spmv( c_mone, A, *x, c_zero, r );      //  r = - A * x
        magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);  //  r = r + b
        H(1,0) = MAGMA_Z_MAKE( magma_dznrm2(dofs, r.val, 1), 0. ); 
                                            //  RNorm = H[1][0] = || r ||
        RNorm = MAGMA_Z_REAL( H(1,0) );
        betanom = fabs(RNorm);  
        #ifdef ENABLE_TIMER
        //Chronometry  
        magma_device_sync(); tempo2=magma_wtime();
        if( solver_par->numiter%iterblock==0 ) {
            printf("   %4d    ||    %e    ||    %.4lf  \n", 
                (solver_par->numiter), betanom, tempo2-tempo1 );
        }
        #endif
        if (  betanom  < r0 ) {
            break;
        } 
    }

    #ifdef ENABLE_TIMER
    double residual;
    magma_zresidual( A, b, *x, &residual );
    printf("#=============================================================#\n");
    printf("# GMRES-(%d) solver summary:\n", restart);
    printf("#    initial residual: %e\n", nom0 );
    printf("#    iterations: %4d\n#    iterative residual: %e\n",
            (solver_par->numiter), betanom );
    printf("#    exact relative residual: %e\n#    runtime: %.4lf sec\n", 
                residual, tempo2-tempo1);
    printf("#=============================================================#\n");
    #endif
  
    solver_par->residual = (double)(betanom);

    // free GPU streams and events
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_event_destroy( event[0] );
    magmablasSetKernelStream(NULL);
    // free pinned memory
    magma_free_pinned( H );
    magma_free_pinned( y );
    magma_free_pinned( HH );
    magma_free_pinned( h1 );
    // free GPU memory
    magma_free(dy); 
    if (dH != NULL ) magma_free(dH); 
    magma_z_vfree(&r);
    magma_z_vfree(&q);

    return MAGMA_SUCCESS;
}   /* magma_zgmres */

