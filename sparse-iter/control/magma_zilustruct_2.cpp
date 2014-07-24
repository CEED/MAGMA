/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from 
//  the IO functions provided by MatrixMarket

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>

#include "magmasparse_z.h"
#include "magma.h"
#include "mmio.h"

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>


using namespace std;


#include <stdlib.h>
#include "mex.h"

/******************************************************************************
 * ILU mex function from MATLAB:
 *
 * [l, u] = ilu_mex(a, level, omega, storage);
 *****************************************************************************/

#define MIN(x,y) ((x)<(y) ? (x) : (y))

void shell_sort(
  const int n,
  int x[]);

void symbolic_ilu(
  const int levinc,
  const int n,
  int *nzl,
  int *nzu,
  const mwIndex ia[], const mwIndex ja[],
  mwIndex ial[], mwIndex jal[],
  mwIndex iau[], mwIndex jau[]);

void numeric_rilu(
  const int n,
  const mwIndex ia[], const mwIndex ja[], const double a[],
  mwIndex ial[], mwIndex jal[], double al[],
  mwIndex iau[], mwIndex jau[], double au[], double omega);

/******************************************************************************
 *
 * MEX function
 *
 *****************************************************************************/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* matrix is stored in CSC format, 0-based */

    int n;
    mwIndex *ia, *ja, *ial, *jal, *iau, *jau;
    double *a, *al, *au;
    int levfill, storage, nzl, nzu, ret;
    double omega;

    if (nrhs < 4)
        mexErrMsgTxt("ilu mex function called with bad number of arguments");

     n = mxGetN(prhs[0]);
    ia = mxGetJc(prhs[0]);
    ja = mxGetIr(prhs[0]);
     a = mxGetPr(prhs[0]);

    levfill = (int)    *mxGetPr(prhs[1]);
    omega   = (double) *mxGetPr(prhs[2]);
    storage = (int)    *mxGetPr(prhs[3]);
    nzl = storage;
    nzu = storage;

    plhs[1] = mxCreateSparse(n, n, storage, 0);
    plhs[0] = mxCreateSparse(n, n, storage, 0);

    /* change order of L and U, since the mex code is row-oriented */
    ial = mxGetJc(plhs[1]);
    jal = mxGetIr(plhs[1]);
     al = mxGetPr(plhs[1]);
    iau = mxGetJc(plhs[0]);
    jau = mxGetIr(plhs[0]);
     au = mxGetPr(plhs[0]);

    /* the following will fail and return to matlab if insufficient storage */
    symbolic_ilu(levfill, n, &nzl, &nzu, ia, ja, ial, jal, iau, jau);

    /* the following assumes that the rows are sorted */
    /* the symbolic routine above assures this */
    numeric_rilu(n, ia, ja, a, ial, jal, al, iau, jau, au, omega);
}

/* shell sort
// stable, so it is fast if already sorted
// sorts x[0:n-1] in place, ascending order.
*/

void shell_sort(
  const int n,
  int x[])
{
    int m, max, j, k, itemp;

    m = n/2;

    while (m > 0) {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] >= x[k])
                    break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }  
        m = m/2;
    }
}

/*
// symbolic level ILU
// factors into separate upper and lower parts
// sorts the entries in each row of A by index
// assumes no zero rows
*/

void symbolic_ilu(
  const int levfill,                 /* level of fill */
  const int n,                       /* order of matrix */
  int *nzl,                          /* input-output */
  int *nzu,                          /* input-output */
  const mwIndex ia[], const mwIndex ja[],    /* input */
  mwIndex ial[], mwIndex jal[],              /* output lower factor structure */
  mwIndex iau[], mwIndex jau[])              /* output upper factor structure */
{
    int i;
    int *lnklst = mxCalloc(n, sizeof(int));
    int *curlev = mxCalloc(n, sizeof(int));
    int *levels = mxCalloc(*nzu, sizeof(int));
    int *iwork = mxCalloc(n, sizeof(int));

    int knzl = 0;
    int knzu = 0;

    ial[0] = 0;
    iau[0] = 0;

    for (i=0; i<n; i++)
    {
        int first, next, j;

        /* copy column indices of row into workspace and sort them */

        int len = ia[i+1] - ia[i];
        next = 0;
        for (j=ia[i]; j<ia[i+1]; j++)
            iwork[next++] = ja[j];
        shell_sort(len, iwork);

        /* construct implied linked list for row */

        first = iwork[0];
        curlev[first] = 0;

        for (j=0; j<=len-2; j++)
        {
            lnklst[iwork[j]] = iwork[j+1];
            curlev[iwork[j]] = 0;
        }

        lnklst[iwork[len-1]] = n;
        curlev[iwork[len-1]] = 0;

        /* merge with rows in U */

        next = first;
        while (next < i)
        {
            int oldlst = next;
            int nxtlst = lnklst[next];
            int row = next;
            int ii;

            /* scan row */

            for (ii=iau[row]+1; ii<iau[row+1]; /*nop*/)
            {
                if (jau[ii] < nxtlst)
                {
                    /* new fill-in */
                    int newlev = curlev[row] + levels[ii] + 1;
                    if (newlev <= levfill)
                    {
                        lnklst[oldlst]  = jau[ii];
                        lnklst[jau[ii]] = nxtlst;
                        oldlst = jau[ii];
                        curlev[jau[ii]] = newlev;
                    }
                    ii++;
                }
                else if (jau[ii] == nxtlst)
                {
            int newlev;
                    oldlst = nxtlst;
                    nxtlst = lnklst[oldlst];
                    newlev = curlev[row] + levels[ii] + 1;
                    curlev[jau[ii]] = MIN(curlev[jau[ii]], newlev);
                    ii++;
                }
                else /* (jau[ii] > nxtlst) */
                {
                    oldlst = nxtlst;
                    nxtlst = lnklst[oldlst];
                }
            }
            next = lnklst[next];
        }
        
        /* gather the pattern into L and U */

        next = first;
        while (next < i)
        {
            if (knzl >= *nzl)
        {
            mexPrintf("ILU: STORAGE parameter value %d too small.\n", *nzl);
                mexErrMsgTxt("Increase STORAGE parameter.");
        }
            jal[knzl++] = next;
            next = lnklst[next];
        }
        ial[i+1] = knzl;

        if (next != i)
        {
        mexErrMsgTxt("ILU structurally singular.\n");
        /*
            assert(knzu < *nzu);
            levels[knzu] = 2*n;
            jau[knzu++] = i;
        */
        }

        while (next < n)
        {
            if (knzu >= *nzu)
        {
            mexPrintf("ILU: STORAGE parameter value %d too small.\n", *nzu);
                mexErrMsgTxt("Increase STORAGE parameter.");
        }
            levels[knzu] = curlev[next];
            jau[knzu++] = next;
            next = lnklst[next];
        }
        iau[i+1] = knzu;
    }

    mxFree(lnklst);
    mxFree(curlev);
    mxFree(levels);
    mxFree(iwork);

    *nzl = knzl;
    *nzu = knzu;

#if 0
    cout << "Actual nnz for ILU: " << *nzl + *nzu << endl;
#endif
}

/*
// assumes diag is first element in U
*/

void numeric_rilu(
  const int n, 
  const mwIndex ia[], const mwIndex ja[], const double a[],
  mwIndex ial[], mwIndex jal[], double al[],
  mwIndex iau[], mwIndex jau[], double au[], double omega)
{
    int i, k, kk, id, idd;
    double mult, modif;

    double *row = mxCalloc(n, sizeof(double));
    int *marker = mxCalloc(n, sizeof(int));

    for (i=0; i<n; i++)
    {
        row[i] = 0.0;
        marker[i] = 0;
    }

    for (i=0; i<n; i++)
    {
        /* scatter row of A */
        for (k=ia[i]; k<ia[i+1]; k++)
            row[ja[k]] = a[k];

        /* scatter data structure of L and U */
        for (k=ial[i]; k<ial[i+1]; k++)
            marker[jal[k]] = 1;
        for (k=iau[i]; k<iau[i+1]; k++)
            marker[jau[k]] = 1;
        
        modif = 0.0;

        /* eliminate the elements in L in order */
        for (k=ial[i]; k<ial[i+1]; k++)
        {
            id = jal[k];
            mult = row[id] / au[iau[id]];
            row[id] = mult;

            for (kk=iau[id]+1; kk<iau[id+1]; kk++)
            {
                idd = jau[kk];
                if (marker[idd])
                    row[idd] -= mult*au[kk];
                else
                    modif -= mult*au[kk];
            }
        }

        /* gather resulting rows in L and U */
        for (k=ial[i]; k<ial[i+1]; k++)
        {
            al[k] = row[jal[k]];
            row[jal[k]] = 0.0;
            marker[jal[k]] = 0;
        }
        for (k=iau[i]; k<iau[i+1]; k++)
        {
            au[k] = row[jau[k]];
            row[jau[k]] = 0.0;
            marker[jau[k]] = 0;
        }

        /* modified */
        au[iau[i]] += omega*modif;
#if 0
        printf("pivot: %e\n",  au[iau[i]]);
#endif
    }

    mxFree(marker);
    mxFree(row);
}


/**
    Purpose
    -------

    This routine computes the fill-in structure of an ILU(levels) factorization
    based on the successive multiplication of upper and lower triangular factors
    using the CUSPARSE library.
    The code is written by Edmond Chow (Georgia Tech).

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix*
                matrix in magma sparse matrix format

    @param
    levels      magma_int_t
                fill in level


    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t 
magma_zilustruct_2( magma_z_sparse_matrix *A, magma_int_t levels ){

    
    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ){

        magma_z_sparse_matrix B, L, U, L_d, U_d, LU_d;

        magma_z_cucsrtranspose( *A, &B );

        magma_z_mconvert( B, &L, Magma_CSR, Magma_CSRL );
        magma_z_mconvert( B, &U, Magma_CSR, Magma_CSRL );


void symbolic_ilu( levels, A.num_rows, &L.nnz, &U.nnz, B.row, B.col, L.row, L.col, U.row, U.col ); 
  const int levfill,                 /* level of fill */
  const int n,                       /* order of matrix */
  int *nzl,                          /* input-output */
  int *nzu,                          /* input-output */
  const mwIndex ia[], const mwIndex ja[],    /* input */
  mwIndex ial[], mwIndex jal[],              /* output lower factor structure */
  mwIndex iau[], mwIndex jau[])              /* output upper factor structure */

        magma_zmalloc_cpu( &L.val, L.nnz );
        magma_zmalloc_cpu( &U.val, U.nnz );

        magma_z_mfree( A );
        magma_z_LUmergein( L, U, A);

        magma_z_mfree( L );
        magma_z_mfree( U );
        magma_z_mfree( B );

        return MAGMA_SUCCESS;
    }
    else{

        magma_z_sparse_matrix hA, CSRCOOA;
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        magma_z_mtransfer( *A, &hA, A->memory_location, Magma_CPU );
        magma_z_mconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSR );

        magma_zilustruct( &CSRCOOA, levels );

        magma_z_mfree( &hA );
        magma_z_mfree( A );
        magma_z_mconvert( CSRCOOA, &hA, Magma_CSR, A_storage );
        magma_z_mtransfer( hA, A, Magma_CPU, A_location );
        magma_z_mfree( &hA );
        magma_z_mfree( &CSRCOOA );    

        return MAGMA_SUCCESS; 
    }
}

