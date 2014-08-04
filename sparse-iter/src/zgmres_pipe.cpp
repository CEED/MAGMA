/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include <sys/time.h>
#include <time.h>

#include "common_magma.h"
#include "../include/magmasparse.h"
#include "trace.h"


#define PRECISION_z

#define RTOLERANCE     10e-10
#define ATOLERANCE     10e-10

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory. 
    This is a GPU implementation of the GMRES method.

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                descriptor for matrix A

    @param
    b           magma_z_vector
                RHS b vector

    @param
    x           magma_z_vector*
                solution approximation

    @param
    solver_par  magma_z_solver_par*
                solver parameters

    @ingroup magmasparse_zgesv
    ********************************************************************/

magma_int_t
magma_zgmres_pipe( magma_z_sparse_matrix A, magma_z_vector b, magma_z_vector *x,  
                   magma_z_solver_par *solver_par ){

#define  q(i)     (q.val + (i)*dofs)
#define hQ(i)     (hQ    + (i)*dofs)
#define  z(i)     (z.val + (i)*dofs)
#define hZ(i)     (hZ    + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*ldh]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) (dH+ (i) + (j)*ldh)

    //Chronometry
    double tempo1, tempo2;

    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE, c_mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    magma_int_t i, j, k, m = 0.0, iter, ldh = solver_par->restart+1, ione = 1, converged = 0;
    double rNorm, RNorm, den, r0 = 0.;

    // CPU workspace
    magmaDoubleComplex *H, *HH, *y, *h1, *hQ, *hZ, *hX, alpha;
    magma_zmalloc_pinned( &HH, (ldh+1)*ldh );
    magma_zmalloc_pinned( &h1, ldh );
    magma_zmalloc_pinned( &H,  (ldh+1)*ldh );
    magma_zmalloc_pinned( &hQ, dofs*(1+ldh) );
    magma_zmalloc_pinned( &hZ, dofs*(1+ldh) );
    magma_zmalloc_pinned( &y, ldh );
    magma_zmalloc_pinned( &hX, dofs );
    
    // GPU workspace
    magma_z_vector r, q, z, zk[2];
    magma_z_vinit( &r  , Magma_DEV, dofs,         c_zero );
    magma_z_vinit( &q  , Magma_DEV, dofs*ldh,     c_zero );
    magma_z_vinit( &z  , Magma_DEV, dofs*(1+ldh), c_zero );
    magma_z_vinit( &zk[0], Magma_DEV, dofs,       c_zero );
    magma_z_vinit( &zk[1], Magma_DEV, dofs,       c_zero );

    // GPU stream
    magma_queue_t stream[3];
    magma_event_t event[1];
    magma_queue_create( &stream[0] );
    magma_queue_create( &stream[1] );
    magma_queue_create( &stream[2] );
    magma_event_create( &event[0] );
    trace_init( 1, 1, 3, (CUstream_st**)stream );

    magmaDoubleComplex *dy;
    if (MAGMA_SUCCESS != magma_zmalloc( &dy, ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;
    magmaDoubleComplex *dH;
    if (MAGMA_SUCCESS != magma_zmalloc( &dH, (ldh+1)*ldh )) 
        return MAGMA_ERR_DEVICE_ALLOC;

    //Chronometry 
    magma_device_sync(); tempo1=magma_wtime();
    //#define ENABLE_TIMER
    #ifdef ENABLE_TIMER
    double t_spmv1, t_spmv = 0.0;
    double t_orth1, t_orth2, t_orth = 0.0, t_cpu = 0.0;
    #endif

    memset( hX, 0, dofs*sizeof(magmaDoubleComplex) );   //  x  = 0

    magmablasSetKernelStream(stream[0]);
    magma_zcopy( dofs, b.val, 1, z(0), 1 );             //  z0 = b
    magma_zgetvector_async(dofs, b.val, 1, hQ(0), 1, stream[1]); //q0 = b
    for (iter = 0; iter<solver_par->maxiter; iter++) 
        {
            for(k=1; k<=(solver_par->restart+1); k++) 
                {
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv1=magma_wtime();
                    #endif
                    //  z[k] = A z[k-1] 
                    zk[0].val = z(k-1);
                    zk[1].val = z(k);
                    magmablasSetKernelStream(stream[0]);
                    trace_gpu_start( 0, 0, "spmv", "spmv" );
                    magma_z_spmv( c_one, A, zk[0], c_zero, zk[1] );                      
                    trace_gpu_end( 0, 0 );
                    //#define ORTHO_Z_ON_CPU
                    #ifdef  ORTHO_Z_ON_CPU
                    magma_zgetvector_async(dofs, z(k), 1, hZ(k), 1, stream[0]);
                    #endif

                    //  compute coefficient H[i][k-1] = q[i] .z(k-1) 
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_spmv += magma_wtime()-t_spmv1;
                    t_orth1 = magma_wtime();
                    #endif
                    magma_queue_sync( stream[1] );
                    trace_cpu_start( 0, "gemv", "gemvH" );
                    lapackf77_zlacpy("F", &dofs, &ione, hQ(k-1), &dofs, hQ(k), &dofs);
                    blasf77_zgemv("Transpose", &dofs, &k, 
                                  &c_one, hQ(0), &dofs,
                                          hQ(k), &ione, 
                                  &c_zero, &H(1,k-1), &ione);
                    trace_cpu_end( 0 );
                    magma_zsetvector_async(k-1, &H(1,k-1), 1, dH(1,k-1), 1, stream[0]);
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_cpu += magma_wtime()-t_orth1;
                    #endif

                    // compute z[k] = A q[k-1]
                    if ( k+1 <= (solver_par->restart+1)  ) {
                        #ifdef  ORTHO_Z_ON_CPU
                        i = k-1;
                        alpha = 1./H(k,k-1);
                        magma_queue_sync( stream[0] );
                        trace_cpu_start( 0, "gemv", "gemvZ" );
                        blasf77_zgemv("NoTrans", &dofs, &i, &c_mone, hZ(1), &dofs, &H(1,k-1), &ione, &c_one, hZ(k), &ione);
                        trace_cpu_end( 0 );
                        #else
                        magmablasSetKernelStream(stream[0]);
                        trace_gpu_start( 0, 0, "gemv", "gemv" );
                        magma_zgemv(MagmaNoTrans, dofs, k-1, c_mone, z(1), dofs, dH(1,k-1), 1, c_one, z(k), 1);
                        trace_gpu_end( 0, 0 );
                        #endif
                    }

                    //  compute H[k+1][k] = sqrt(q[k-1] . q[k-1]) 
                    H(k,k-1) -= magma_cblas_zdotc( k-1, &H(1,k-1), 1, &H(1,k-1), 1 );
                    H(k,k-1)  = MAGMA_Z_MAKE( sqrt( MAGMA_Z_REAL(H(k,k-1))), 0 );
                    magma_zsetvector_async(1, &H(1,k), 1, dH(1,k), 1, stream[0]);
                    if (k == 1) {
                        if (iter == 0) {
                            r0 = MAGMA_Z_REAL(H(k,k-1));
                            if ((r0 *= solver_par->epsilon) < ATOLERANCE)
                                r0 = ATOLERANCE;
                        }
                        RNorm = MAGMA_Z_REAL( H(1,0) );
                        #ifdef ENABLE_TIMER
                        magma_device_sync(); tempo2=magma_wtime();
                        printf("Iteration: %4d  Norm: %e  Time: %.2lf, Orth: %.2lf, %.2lf on CPU (%.2lf%%) SpMV: %.2lf (%.2lf%%)\n", iter+1, RNorm*RNorm, tempo2-tempo1, 
                               t_orth, t_cpu, t_orth/(tempo2-tempo1), t_spmv,t_spmv/(tempo2-tempo1));
                        #endif
                        if (fabs(RNorm*RNorm) < r0) {
                            magma_device_sync(); tempo2=magma_wtime();
                            printf( "\n      Number of GMRES restarts: %d\n", iter);
                            printf( "      Time: %.2lf\n\n",tempo2-tempo1 );
                            converged = 1;
                            break; 
                        }
                    }

                    // compute z[k] = A q[k-1]
                    if ( k+1 <= (solver_par->restart+1)  ) {
                         #ifdef  ORTHO_Z_ON_CPU
                         blasf77_zscal( &dofs, &alpha, hZ(k), &ione );                     
                         magma_zsetvector_async(dofs, hZ(k), 1, z(k), 1, stream[0]);
                         lapackf77_zlacpy("F", &dofs, &ione, hZ(k), &dofs, hQ(k), &dofs);
                         #else
                         magmablasSetKernelStream(stream[0]);
                         trace_gpu_start( 0, 0, "gemv", "gemv" );
                         magma_zscal(dofs, 1./H(k,k-1), z(k), 1);                     
                         trace_gpu_end( 0, 0 );

                         magma_event_record( event[0], stream[0] );
                         magma_queue_wait_event( stream[1], event[0] );
                         trace_gpu_start( 0, 0, "comm", "get" );
                         magma_zgetvector_async(dofs, z(k), 1, hQ(k), 1, stream[1]);
                         trace_gpu_end( 0, 0 );
                         #endif
                    }

                    // orthogonzile q[k-1] 
                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth2 = magma_wtime();
                    #endif
                    i = k-1;
                    alpha = 1./H(k,k-1);
                    trace_cpu_start( 0, "gemv", "gemvQ" );
                    blasf77_zgemv("NoTrans", &dofs, &i, &c_mone, hQ(0), &dofs, &H(1,k-1), &ione, &c_one, hQ(k-1), &ione);
                    blasf77_zscal( &dofs, &alpha, hQ(k-1), &ione );                     
                    trace_cpu_end( 0 );
                    #ifdef ENABLE_TIMER
                    t_cpu += magma_wtime()-t_orth2;
                    #endif

                    #ifdef ENABLE_TIMER
                    magma_device_sync(); t_orth += magma_wtime()-t_orth1;
                    #endif
                }
            if (converged == 1 ) break;

            for(k=1; k<=(solver_par->restart); k++) 
                {
                    /*     Minimization of  || b-Ax ||  in H_k       */ 
                    for (i=1; i<=k; i++) {
                        HH(k,i) = magma_cblas_zdotc( i+1, &H(1,k), 1, &H(1,i), 1 );
                    }
                    
                    h1[k] = H(1,k)*H(1,0);
                    
                    if (k != 1)
                        for (i=1; i<k; i++) {
                            for (m=i+1; m<k; m++) {
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
                            for (j=i+1; j<=k; j++) {
                                y[i] -= y[j] * HH(j,i);
                            }
                        }
                    
                    m = k;
                    
                    rNorm = fabs(MAGMA_Z_REAL(H(k+1,k)));
                }
            
            /*   Update the current approximation: x += Q y  */
            blasf77_zgemv("NoTrans", &dofs, &m, &c_one, hQ(0), &dofs, y+1, &ione, &c_one, hX, &ione);
            magma_zsetvector_async(dofs, hX, 1, x->val, 1, stream[0]);

            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv1=magma_wtime();
            #endif
            magmablasSetKernelStream(stream[0]);
            magma_z_spmv( c_mone, A, *x, c_zero, r );                  //  r = - A * x
            #ifdef ENABLE_TIMER
            magma_device_sync(); t_spmv += magma_wtime() - t_spmv1;
            #endif

            magma_zaxpy(dofs, c_one, b.val, 1, r.val, 1);   //  r = r + b
            magma_zcopy( dofs, r.val, 1, z(0), 1 );         //  z0 = b

            magma_event_record( event[0], stream[0] );
            magma_queue_wait_event( stream[1], event[0] );
            magma_zgetvector_async(dofs, z(0), 1, hQ(0), 1, stream[1]);
        }
    
    if (converged == 0 ) {
        magma_device_sync(); tempo2=magma_wtime();
        printf( "\n    ** Failed to converge within %d restart **\n",solver_par->maxiter );
        printf( "      Number of GMRES restarts: %d\n", iter);
        printf( "      Time: %.2lf\n\n",tempo2-tempo1 );
    }
    
    if (solver_par->epsilon == RTOLERANCE) {
        magma_z_spmv( c_one, A, *x, c_zero, r );                       // q_t = A x
        magma_zaxpy(dofs,  c_mone, b.val, 1, r.val, 1);                // r = r - b
        den = magma_dznrm2(dofs, r.val, 1);                            // den = || r ||
        printf( "      || r_N ||   = %f\n", den);
        solver_par->residual = (double)(den);
    }
    solver_par->numiter = iter;
    trace_finalize( "zgmres_pipe.svg","trace.css" );

    // free GPU streams and events
    magma_queue_destroy( stream[0] );
    magma_queue_destroy( stream[1] );
    magma_queue_destroy( stream[2] );
    magma_event_destroy( event[0] );

    // free CPU memory
    magma_free_pinned( hQ );
    magma_free_pinned( hZ );
    magma_free_pinned( hX );
    magma_free_pinned( H );
    magma_free_pinned( y );
    magma_free_pinned( HH );
    magma_free_pinned( h1 );

    // free GPU memory
    magma_free(dy); 
    magma_free(dH);  
    magma_z_vfree(&r); 
    magma_z_vfree(&q);
    magma_z_vfree(&z);

    return MAGMA_SUCCESS;
}   /* magma_zgmres_pipe */

