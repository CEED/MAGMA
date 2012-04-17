/*  -- MAGMA (version 1.1) --
    Univ. of Tennessee, Knoxville
    Univ. of California, Berkeley
    Univ. of Colorado, Denver
    November 2011

    @author Raffaele Solca
    @precisions normal d -> s
*/
#define N_MAX_GPU 8

#include "common_magma.h"
#include <cblas.h>

#define Q(ix, iy) (q + (ix) + ldq * (iy))

#define dQ2(id) (dwork[id])
#define dS(id, ii) (dwork[id] + n2*n2_loc + ii * n2*nb)
#define dQ(id, ii) (dwork[id] + n2*n2_loc + 2 * n2*nb + ii * n2_loc*nb)


extern "C"{
    int magma_get_dlaex3_m_k() { return 512;}
    int magma_get_dlaex3_m_nb() { return 1024;}
    double lapackf77_dlamc3(double* a,double* b);
    void lapackf77_dlamrg(magma_int_t* n1, magma_int_t* n2, double* a, magma_int_t* dtrd1, magma_int_t* dtrd2, magma_int_t* index);
    void lapackf77_dlaed4(magma_int_t* n, magma_int_t* i, double* d, double* z,
                          double* delta, double* rho, double* dlam, magma_int_t* info);
    magma_int_t magma_dlaex3(magma_int_t k, magma_int_t n, magma_int_t n1, double* d,
                             double* q, magma_int_t ldq, double rho,
                             double* dlamda, double* q2, magma_int_t* indx,
                             magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
                             double* dwork,
                             char range, double vl, double vu, magma_int_t il, magma_int_t iu,
                             magma_int_t* info );
}

extern"C"{
    void dvrange(magma_int_t k, double *d, magma_int_t *il, magma_int_t *iu, double vl, double vu)
    {
        magma_int_t i;

        *il=1;
        *iu=k;
        for (i = 0; i < k; ++i){
            if (d[i] > vu){
                *iu = i;
                break;
            }
            else if (d[i] < vl)
                ++il;
        }
        return;
    }

    void dirange(magma_int_t k, magma_int_t* indxq, magma_int_t *iil, magma_int_t *iiu, magma_int_t il, magma_int_t iu)
    {
        magma_int_t i;

        *iil = 1;
        *iiu = 0;
        for (i = il; i<=iu; ++i)
            if (indxq[i-1]<=k){
                *iil = indxq[i-1];
                break;
            }
        for (i = iu; i>=il; --i)
            if (indxq[i-1]<=k){
                *iiu = indxq[i-1];
                break;
            }
        return;
    }
}

extern "C" magma_int_t
magma_dlaex3_m(magma_int_t nrgpu,
               magma_int_t k, magma_int_t n, magma_int_t n1, double* d,
               double* q, magma_int_t ldq, double rho,
               double* dlamda, double* q2, magma_int_t* indx,
               magma_int_t* ctot, double* w, double* s, magma_int_t* indxq,
               double** dwork, cudaStream_t stream[N_MAX_GPU][2],
               char range, double vl, double vu, magma_int_t il, magma_int_t iu,
               magma_int_t* info )
{
    /*
     Purpose
     =======

     DLAEX3 finds the roots of the secular equation, as defined by the
     values in D, W, and RHO, between 1 and K.  It makes the
     appropriate calls to DLAED4 and then updates the eigenvectors by
     multiplying the matrix of eigenvectors of the pair of eigensystems
     being combined by the matrix of eigenvectors of the K-by-K system
     which is solved here.

     It is used in the last step when only a part of the eigenvectors
     is required.
     It compute only the required part of the eigenvectors and the rest
     is not used.

     This code makes very mild assumptions about floating point
     arithmetic. It will work on machines with a guard digit in
     add/subtract, or on those binary machines without guard digits
     which subtract like the Cray X-MP, Cray Y-MP, Cray C-90, or Cray-2.
     It could conceivably fail on hexadecimal or decimal machines
     without guard digits, but we know of none.

     Arguments
     =========

     K       (input) INTEGER
     The number of terms in the rational function to be solved by
     DLAED4.  K >= 0.

     N       (input) INTEGER
     The number of rows and columns in the Q matrix.
     N >= K (deflation may result in N>K).

     N1      (input) INTEGER
     The location of the last eigenvalue in the leading submatrix.
     min(1,N) <= N1 <= N/2.

     D       (output) DOUBLE PRECISION array, dimension (N)
     D(I) contains the updated eigenvalues for
     1 <= I <= K.

     Q       (output) DOUBLE PRECISION array, dimension (LDQ,N)
     Initially the first K columns are used as workspace.
     On output the columns ??? to ??? contain
     the updated eigenvectors.

     LDQ     (input) INTEGER
     The leading dimension of the array Q.  LDQ >= max(1,N).

     RHO     (input) DOUBLE PRECISION
     The value of the parameter in the rank one update equation.
     RHO >= 0 required.

     DLAMDA  (input/output) DOUBLE PRECISION array, dimension (K)
     The first K elements of this array contain the old roots
     of the deflated updating problem.  These are the poles
     of the secular equation. May be changed on output by
     having lowest order bit set to zero on Cray X-MP, Cray Y-MP,
     Cray-2, or Cray C-90, as described above.

     Q2      (input) DOUBLE PRECISION array, dimension (LDQ2, N)
     The first K columns of this matrix contain the non-deflated
     eigenvectors for the split problem.

     INDX    (input) INTEGER array, dimension (N)
     The permutation used to arrange the columns of the deflated
     Q matrix into three groups (see DLAED2).
     The rows of the eigenvectors found by DLAED4 must be likewise
     permuted before the matrix multiply can take place.

     CTOT    (input) INTEGER array, dimension (4)
     A count of the total number of the various types of columns
     in Q, as described in INDX.  The fourth column type is any
     column which has been deflated.

     W       (input/output) DOUBLE PRECISION array, dimension (K)
     The first K elements of this array contain the components
     of the deflation-adjusted updating vector. Destroyed on
     output.

     S       (workspace) DOUBLE PRECISION array, dimension (N1 + 1)*K
     Will contain the eigenvectors of the repaired matrix which
     will be multiplied by the previously accumulated eigenvectors
     to update the system.

     INDXQ  (output) INTEGER array, dimension (N)
     On exit, the permutation which will reintegrate the
     subproblems back into sorted order,
     i.e. D( INDXQ( I = 1, N ) ) will be in ascending order.

     DWORK  (devices workspaces) DOUBLE PRECISION array of arrays,
     dimension NRGPU.
     if NRGPU = 1 the dimension of the first workspace
     should be (3*N*N/2+3*N)
     otherwise the NRGPU workspaces should have the size
     ceil((N-N1) * (N-N1) / floor(nrgpu/2)) +
     NB * ((N-N1) + (N-N1) / floor(nrgpu/2))

     STREAM (device stream) cudaStream_t array,
     dimension (N_MAX_GPU,2)

     INFO    (output) INTEGER
     = 0:  successful exit.
     < 0:  if INFO = -i, the i-th argument had an illegal value.
     > 0:  if INFO = 1, an eigenvalue did not converge

     Further Details
     ===============

     Based on contributions by
     Jeff Rutter, Computer Science Division, University of California
     at Berkeley, USA
     Modified by Francoise Tisseur, University of Tennessee.

     =====================================================================
     */
    if (nrgpu==1){
        cudaSetDevice(0);
        magma_dlaex3(k, n, n1, d, q, ldq, rho,
                     dlamda, q2, indx, ctot, w, s, indxq,
                     *dwork, range, vl, vu, il, iu, info );
        return MAGMA_SUCCESS;
    }
    double d_one  = 1.;
    double d_zero = 0.;
    magma_int_t ione = 1;
    magma_int_t imone = -1;
    char range_[] = {range, 0};

    magma_int_t iil, iiu, rk;
    magma_int_t n1_loc, n2_loc, ib, nb, ib2, igpu;
    magma_int_t ni_loc[N_MAX_GPU];

    magma_int_t i,ind,iq2,j,n12,n2,n23,tmp,lq2;
    double temp;
    magma_int_t alleig, valeig, indeig;

    alleig = lapackf77_lsame(range_, "A");
    valeig = lapackf77_lsame(range_, "V");
    indeig = lapackf77_lsame(range_, "I");

    *info = 0;

    if(k < 0)
        *info=-1;
    else if(n < k)
        *info=-2;
    else if(ldq < max(1,n))
        *info=-6;
    else if (! (alleig || valeig || indeig))
        *info = -15;
    else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -17;
        }
        else if (indeig) {
            if (il < 1 || il > max(1,n))
                *info = -18;
            else if (iu < min(n,il) || iu > n)
                *info = -19;
        }
    }


    if(*info != 0){
        magma_xerbla(__func__, -(*info));
        return MAGMA_ERR_ILLEGAL_VALUE;
    }

    // Quick return if possible
    if(k == 0)
        return MAGMA_SUCCESS;
    /*
     Modify values DLAMDA(i) to make sure all DLAMDA(i)-DLAMDA(j) can
     be computed with high relative accuracy (barring over/underflow).
     This is a problem on machines without a guard digit in
     add/subtract (Cray XMP, Cray YMP, Cray C 90 and Cray 2).
     The following code replaces DLAMDA(I) by 2*DLAMDA(I)-DLAMDA(I),
     which on any of these machines zeros out the bottommost
     bit of DLAMDA(I) if it is 1; this makes the subsequent
     subtractions DLAMDA(I)-DLAMDA(J) unproblematic when cancellation
     occurs. On binary machines with a guard digit (almost all
     machines) it does not change DLAMDA(I) at all. On hexadecimal
     and decimal machines with a guard digit, it slightly
     changes the bottommost bits of DLAMDA(I). It does not account
     for hexadecimal or decimal machines without guard digits
     (we know of none). We use a subroutine call to compute
     2*DLAMBDA(I) to prevent optimizing compilers from eliminating
     this code.*/
    double *dwS[2][N_MAX_GPU], *dwQ[2][N_MAX_GPU], *dwQ2[N_MAX_GPU];
//#define CHECK_CPU
#ifdef CHECK_CPU
    double *hwS[2][N_MAX_GPU], *hwQ[2][N_MAX_GPU], *hwQ2[N_MAX_GPU];
#endif
    n2 = n - n1;

    n12 = ctot[0] + ctot[1];
    n23 = ctot[1] + ctot[2];

    iq2 = n1 * n12;
    lq2 = iq2 + n2 * n23;

    n1_loc = (n1-1) / (nrgpu/2) + 1;
    n2_loc = (n2-1) / (nrgpu/2) + 1;

    nb = magma_get_dlaex3_m_nb();

    if (n1 >= magma_get_dlaex3_m_k()){
        for (igpu = 0; igpu < nrgpu; ++igpu){
#ifdef CHECK_CPU
            cudaMallocHost(&(hwS[0][igpu]),n2*nb*sizeof(double) );
            cudaMallocHost(&(hwS[1][igpu]),n2*nb*sizeof(double) );
            cudaMallocHost(&(hwQ2[igpu]),n2*n2_loc*sizeof(double) );
            cudaMallocHost(&(hwQ[0][igpu]),n2_loc*nb*sizeof(double) );
            cudaMallocHost(&(hwQ[1][igpu]),n2_loc*nb*sizeof(double) );
#endif
/*            cudaSetDevice(igpu);
            cudaMalloc(&(dwS[0][igpu]),n2*nb*sizeof(double) );
            cudaMalloc(&(dwS[1][igpu]),n2*nb*sizeof(double) );
            cudaMalloc(&(dwQ2[igpu]),n2*n2_loc*sizeof(double) );
            cudaMalloc(&(dwQ[0][igpu]),n2_loc*nb*sizeof(double) );
            cudaMalloc(&(dwQ[1][igpu]),n2_loc*nb*sizeof(double) );
*/        }
        for (igpu = 0; igpu < nrgpu-1; igpu += 2){
            ni_loc[igpu] = min(n1_loc, n1 - igpu/2 * n1_loc);
#ifdef CHECK_CPU
            lapackf77_dlacpy("A", &ni_loc[igpu], &n12, q2+n1_loc*(igpu/2), &n1, hQ2(igpu), &n1_loc);
#endif
            cudaSetDevice(igpu);
            cudaMemcpy2DAsync(dQ2(igpu), sizeof(double)*n1_loc, q2+n1_loc*(igpu/2), sizeof(double)*n1,
                              sizeof(double)*ni_loc[igpu], n12, cudaMemcpyHostToDevice, stream[igpu][0]);
            ni_loc[igpu+1] = min(n2_loc, n2 - igpu/2 * n2_loc);
#ifdef CHECK_CPU
            lapackf77_dlacpy("A", &ni_loc[igpu+1], &n23, q2+iq2+n2_loc*(igpu/2), &n2, hQ2(igpu+1), &n2_loc);
#endif
            cudaSetDevice(igpu+1);
            cudaMemcpy2DAsync(dQ2(igpu+1), sizeof(double)*n2_loc, q2+iq2+n2_loc*(igpu/2), sizeof(double)*n2,
                              sizeof(double)*ni_loc[igpu+1], n23, cudaMemcpyHostToDevice, stream[igpu+1][0]);
        }
    }

    //#define ENABLE_TIMER
#ifdef ENABLE_TIMER
    magma_timestr_t start, end;
#endif

    for(i = 0; i < k; ++i)
        dlamda[i]=lapackf77_dlamc3(&dlamda[i], &dlamda[i]) - dlamda[i];

#ifdef ENABLE_TIMER
    start = get_current_time();
#endif

#pragma omp parallel for
    for(j = 0; j < k; ++j){
        magma_int_t tmpp=j+1;
        magma_int_t iinfo = 0;
        lapackf77_dlaed4(&k, &tmpp, dlamda, w, Q(0,j), &rho, &d[j], &iinfo);
        // If the zero finder fails, the computation is terminated.
        if(iinfo != 0)
            *info=iinfo;
    }
    if(*info != 0)
        return MAGMA_SUCCESS;

#ifdef ENABLE_TIMER
    end = get_current_time();

    printf("for dlaed4 = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    //Prepare the INDXQ sorting permutation.
    magma_int_t nk = n - k;
    lapackf77_dlamrg( &k, &nk, d, &ione , &imone, indxq);

    //compute the lower and upper bound of the non-deflated eigenvectors
    if (valeig)
        dvrange(k, d, &iil, &iiu, vl, vu);
    else if (indeig)
        dirange(k, indxq, &iil, &iiu, il, iu);
    else {
        iil = 1;
        iiu = k;
    }
    rk = iiu - iil + 1;

    if (k == 2){

        for(j = 0; j < k; ++j){
            w[0] = *Q(0,j);
            w[1] = *Q(1,j);

            i = indx[0] - 1;
            *Q(0,j) = w[i];
            i = indx[1] - 1;
            *Q(1,j) = w[i];
        }

    }
    else if(k != 1){

        // Compute updated W.
        blasf77_dcopy( &k, w, &ione, s, &ione);

        // Initialize W(I) = Q(I,I)
        tmp = ldq + 1;
        blasf77_dcopy( &k, q, &tmp, w, &ione);
#ifdef ENABLE_TIMER
        start = get_current_time();
#endif
#pragma omp parallel for
        for(magma_int_t ii = 0; ii < k; ++ii)
            for(magma_int_t jj = 0; jj < k; ++jj){
                if(ii != jj)
                    w[ii] = w[ii] * ( *Q(ii, jj) / ( dlamda[ii] - dlamda[jj] ) );
            }

#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("for j for i divided in two parts = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

        for(i = 0; i < k; ++i)
            w[i] = copysign( sqrt( -w[i] ), s[i]);

#ifdef ENABLE_TIMER
        start = get_current_time();
#endif

        // Compute eigenvectors of the modified rank-1 modification.
        if (k > 256)
            for(j = iil-1; j < iiu; ++j){
#pragma omp parallel for
                for(magma_int_t ii = 0; ii < k; ++ii)
                    s[ii] = w[ii] / *Q(ii,j);
                temp = cblas_dnrm2( k, s, 1);
#pragma omp parallel for
                for(magma_int_t ii = 0; ii < k; ++ii){
                    magma_int_t iii = indx[ii] - 1;
                    *Q(ii,j) = s[iii] / temp;
                }
            }
        else
            for(j = iil-1; j < iiu; ++j){
                for(i = 0; i < k; ++i)
                    s[i] = w[i] / *Q(i,j);
                temp = cblas_dnrm2( k, s, 1);
                for(i = 0; i < k; ++i){
                    magma_int_t iii = indx[i] - 1;
                    *Q(i,j) = s[iii] / temp;
                }
            }
#ifdef ENABLE_TIMER
        end = get_current_time();

        printf("for j (2*for i) = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif
    }
    // Compute the updated eigenvectors.

#ifdef ENABLE_TIMER
    start = get_current_time();
#endif

    if(rk > 0){
        if (n1 < magma_get_dlaex3_m_k()){
            // stay on the CPU
            if( n23 != 0 ){
                lapackf77_dlacpy("A", &n23, &rk, Q(ctot[0],iil-1), &ldq, s, &n23);
                blasf77_dgemm("N", "N", &n2, &rk, &n23, &d_one, &q2[iq2], &n2,
                              s, &n23, &d_zero, Q(n1,iil-1), &ldq );
            }
            else
                lapackf77_dlaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);

            if( n12 != 0 ) {
                lapackf77_dlacpy("A", &n12, &rk, Q(0,iil-1), &ldq, s, &n12);
                blasf77_dgemm("N", "N", &n1, &rk, &n12, &d_one, q2, &n1,
                              s, &n12, &d_zero, Q(0,iil-1), &ldq);
            }
            else
                lapackf77_dlaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
        }
        else {
            //use the gpus
            ib = min(nb, rk);
            for (igpu = 0; igpu < nrgpu-1; igpu += 2){
                if (n23 != 0) {
                    cudaSetDevice(igpu+1);
                    cudaMemcpy2DAsync(dS(igpu+1,0), sizeof(double)*n23, Q(ctot[0],iil-1), sizeof(double)*ldq,
                                      sizeof(double)*n23, ib, cudaMemcpyHostToDevice, stream[igpu+1][0]);
                }
                if (n12 != 0) {
                    cudaSetDevice(igpu);
                    cudaMemcpy2DAsync(dS(igpu,0), sizeof(double)*n12, Q(0,iil-1), sizeof(double)*ldq,
                                      sizeof(double)*n12, ib, cudaMemcpyHostToDevice, stream[igpu][0]);
                }
            }

            for (i = 0; i<rk; i+=nb){
                ib = min(nb, rk - i);
                ind = (i/nb)%2;
                if (i+nb<rk){
                    ib2 = min(nb, rk - i - nb);
                    for (igpu = 0; igpu < nrgpu-1; igpu += 2){
                        if (n23 != 0) {
                            cudaSetDevice(igpu+1);
                            cudaMemcpy2DAsync(dS(igpu+1,(ind+1)%2), sizeof(double)*n23, Q(ctot[0],iil-1+i+nb), sizeof(double)*ldq,
                                              sizeof(double)*n23, ib2, cudaMemcpyHostToDevice, stream[igpu+1][(ind+1)%2]);
                        }
                        if (n12 != 0) {
                            cudaSetDevice(igpu);
                            cudaMemcpy2DAsync(dS(igpu,(ind+1)%2), sizeof(double)*n12, Q(0,iil-1+i+nb), sizeof(double)*ldq,
                                              sizeof(double)*n12, ib2, cudaMemcpyHostToDevice, stream[igpu][(ind+1)%2]);
                        }
                    }
                }

                // Ensure that the data is copied on gpu since we will overwrite it.
                for (igpu = 0; igpu < nrgpu-1; igpu += 2){
                    if (n23 != 0) {
#ifdef CHECK_CPU
                        lapackf77_dlacpy("A", &n23, &ib, Q(ctot[0],iil-1+i), &ldq, hS(igpu+1,ind), &n23);
#endif
                        cudaSetDevice(igpu+1);
                        //cudaMemcpy2DAsync(dS(igpu+1,ind), sizeof(double)*n23, Q(ctot[0],iil-1+i), sizeof(double)*ldq,
                        //                  sizeof(double)*n23, ib, cudaMemcpyHostToDevice, stream[igpu+1][ind]);
                        cudaStreamSynchronize(stream[igpu+1][ind]);
                    }
                    if (n12 != 0) {
#ifdef CHECK_CPU
                        lapackf77_dlacpy("A", &n12, &ib, Q(0,iil-1+i), &ldq, hS(igpu,ind), &n12);
#endif
                        cudaSetDevice(igpu);
                        //cudaMemcpy2DAsync(dS(igpu,ind), sizeof(double)*n12, Q(0,iil-1+i), sizeof(double)*ldq,
                        //                  sizeof(double)*n12, ib, cudaMemcpyHostToDevice, stream[igpu][ind]);
                        cudaStreamSynchronize(stream[igpu][ind]);
                    }

                }
                for (igpu = 0; igpu < nrgpu-1; igpu += 2){
                    if (n23 != 0) {
#ifdef CHECK_CPU
                        blasf77_dgemm("N", "N", &ni_loc[igpu+1], &ib, &n23, &d_one, hQ2(igpu+1), &n2_loc,
                                      hS(igpu+1,ind), &n23, &d_zero, hQ(igpu+1, ind), &n2_loc);
#endif
                        cudaSetDevice(igpu+1);
                        cublasSetKernelStream(stream[igpu+1][ind]);
                        cublasDgemm('N', 'N', ni_loc[igpu+1], ib, n23, d_one, dQ2(igpu+1), n2_loc,
                                    dS(igpu+1, ind), n23, d_zero, dQ(igpu+1, ind), n2_loc);
#ifdef CHECK_CPU
                        printf("norm Q %d: %f\n", igpu+1, cpu_gpu_ddiff(ni_loc[igpu+1], ib, hQ(igpu+1, ind), n2_loc, dQ(igpu+1, ind), n2_loc));
#endif
                    }
                    if (n12 != 0) {
#ifdef CHECK_CPU
                        blasf77_dgemm("N", "N", &ni_loc[igpu], &ib, &n12, &d_one, hQ2(igpu), &n1_loc,
                                      hS(igpu,ind%2), &n12, &d_zero, hQ(igpu, ind%2), &n1_loc);
#endif
                        cudaSetDevice(igpu);
                        cublasSetKernelStream(stream[igpu][ind]);
                        cublasDgemm('N', 'N', ni_loc[igpu], ib, n12, d_one, dQ2(igpu), n1_loc,
                                    dS(igpu, ind), n12, d_zero, dQ(igpu, ind), n1_loc);
#ifdef CHECK_CPU
                        printf("norm Q %d: %f\n", igpu, cpu_gpu_ddiff(ni_loc[igpu], ib, hQ(igpu, ind), n1_loc, dQ(igpu, ind), n1_loc));
#endif
                    }
                }
                for (igpu = 0; igpu < nrgpu-1; igpu += 2){
                    if (n23 != 0) {
                        cudaSetDevice(igpu+1);
                        cudaMemcpy2DAsync(Q(n1+n2_loc*(igpu/2),iil-1+i), sizeof(double)*ldq, dQ(igpu+1, ind), sizeof(double)*n2_loc,
                                          sizeof(double)*ni_loc[igpu+1], ib, cudaMemcpyDeviceToHost, stream[igpu+1][ind]);
//                        lapackf77_dlacpy("A", &ni_loc[igpu+1], &ib, hQ(igpu+1, ind%2), &n2_loc, Q(n1+n2_loc*(igpu/2),iil-1+i), &ldq);
                    }
                    if (n12 != 0) {
                        cudaSetDevice(igpu);
                        cudaMemcpy2DAsync(Q(n1_loc*(igpu/2),iil-1+i), sizeof(double)*ldq, dQ(igpu, ind), sizeof(double)*n1_loc,
                                          sizeof(double)*ni_loc[igpu], ib, cudaMemcpyDeviceToHost, stream[igpu][ind]);
//                        lapackf77_dlacpy("A", &ni_loc[igpu], &ib, hQ(igpu, ind%2), &n1_loc, Q(n1_loc*(igpu/2),iil-1+i), &ldq);
                    }
                }
            }
            for (igpu = 0; igpu < nrgpu; ++igpu){
#ifdef CHECK_CPU
                cudaFreeHost(hwS[1][igpu]);
                cudaFreeHost(hwS[0][igpu]);
                cudaFreeHost(hwQ2[igpu]);
                cudaFreeHost(hwQ[1][igpu]);
                cudaFreeHost(hwQ[0][igpu]);
#endif
                cudaSetDevice(igpu);
                cublasSetKernelStream(NULL);
                cudaStreamSynchronize(stream[igpu][0]);
                cudaStreamSynchronize(stream[igpu][1]);
/*                cudaFree(dwS[0][igpu]);
                cudaFree(dwS[1][igpu]);
                cudaFree(dwQ2[igpu]);
                cudaFree(dwQ[0][igpu]);
                cudaFree(dwQ[1][igpu]);
*/            }
            if( n23 == 0 )
                lapackf77_dlaset("A", &n2, &rk, &d_zero, &d_zero, Q(n1,iil-1), &ldq);

            if( n12 == 0 )
                lapackf77_dlaset("A", &n1, &rk, &d_zero, &d_zero, Q(0,iil-1), &ldq);
        }
    }
#ifdef ENABLE_TIMER
    end = get_current_time();

    printf("gemms = %6.2f\n", GetTimerValue(start,end)/1000.);
#endif

    return MAGMA_SUCCESS;
} /*magma_dlaed3_m*/
