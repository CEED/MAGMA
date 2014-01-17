/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#include <fstream>
#include <stdlib.h>
#include <string>
#include <sstream>
#include <iostream>
#include <ostream>
#include <assert.h>
#include <stdio.h>
#include "../include/magmasparse_z.h"
#include "../../include/magma.h"
#include "../include/mmio.h"



using namespace std;

/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Transposes a matrix stored in CSR format.


    Arguments
    =========

    magma_int_t n_rows                  number of rows in input matrix
    magma_int_t n_cols                  number of columns in input matrix
    magma_int_t nnz                     number of nonzeros in input matrix
    magmaDoubleComplex *val              value array of input matrix 
    magma_int_t *row                     row pointer of input matrix
    magma_int_t *col                     column indices of input matrix 
    magma_int_t *new_n_rows              number of rows in transposed matrix
    magma_int_t *new_n_cols              number of columns in transposed matrix
    magma_int_t *new_nnz                 number of nonzeros in transposed matrix
    magmaDoubleComplex **new_val         value array of transposed matrix 
    magma_int_t **new_row                row pointer of transposed matrix
    magma_int_t **new_col                column indices of transposed matrix

    ========================================================================  */

magma_int_t z_transpose_csr(    magma_int_t n_rows, 
                                magma_int_t n_cols, m
                                agma_int_t nnz, 
                                magmaDoubleComplex *val, 
                                magma_int_t *row, 
                                magma_int_t *col, 
                                magma_int_t *new_n_rows, 
                                magma_int_t *new_n_cols, 
                                magma_int_t *new_nnz, 
                                magmaDoubleComplex **new_val, 
                                magma_int_t **new_row, 
                                magma_int_t **new_col ){



  nnz = row[n_rows];
  *new_n_rows = n_cols;
  *new_n_cols = n_rows;
  *new_nnz = nnz;

  magmaDoubleComplex ** valtemp;
  magma_int_t ** coltemp;
  valtemp =(magmaDoubleComplex**)malloc((n_rows)*sizeof(magmaDoubleComplex*));
  coltemp =(magma_int_t**)malloc((n_rows)*sizeof(magma_int_t*));

  // temporary 2-dimensional arrays valtemp/coltemp 
  // where val[i] is the array with the values of the i-th column of the matrix
  magma_int_t nnztemp[n_rows];
  for( magma_int_t i=0; i<n_rows; i++ )
    nnztemp[i]=0;
  for( magma_int_t i=0; i<nnz; i++ )
    nnztemp[col[i]]++;    

  for( magma_int_t i=0; i<n_rows; i++ ){
    valtemp[i] = 
        (magmaDoubleComplex*)malloc((nnztemp[i])*sizeof(magmaDoubleComplex));
    coltemp[i] = (magma_int_t*)malloc(nnztemp[i]*sizeof(magma_int_t));
  }

  for( magma_int_t i=0; i<n_rows; i++ )
    nnztemp[i]=0;

  for( magma_int_t j=0; j<n_rows; j++ ){
    for( magma_int_t i=row[j]; i<row[j+1]; i++ ){
      valtemp[col[i]][nnztemp[col[i]]]=val[i];
      coltemp[col[i]][nnztemp[col[i]]]=j;
      nnztemp[col[i]]++;    
    }
  }

  //csr structure for transposed matrix
  *new_val = new magmaDoubleComplex[nnz];
  *new_row = new magma_int_t[n_rows+1];
  *new_col = new magma_int_t[nnz];

  //fill the transposed csr structure
  magma_int_t nnztmp=0;
  (*new_row)[0]=0;
  for( magma_int_t j=0; j<n_rows; j++ ){
    for( magma_int_t i=0; i<nnztemp[j]; i++ ){
      (*new_val)[nnztmp]=valtemp[j][i];
      (*new_col)[nnztmp]=coltemp[j][i];
      nnztmp++;
    }
    (*new_row)[j+1]=nnztmp;
  }
//usually the temporary memory should be freed afterwards
//however, it does not work
/*
  for( magma_int_t j=0; j<n_rows; j++ ){
    free(valtemp[j]);
    free(coltemp[j]);
  }
  free(valtemp);free(coltemp);
      printf("check9\n");
    fflush(stdout);
*/
}

magma_int_t 
magma_z_mtranspose( magma_z_sparse_matrix A, magma_z_sparse_matrix *B ){

    if( A.memory_location == Magma_CPU ){
        if( A.storage_type == Magma_CSR ){
            z_transpose_csr( A.num_rows, A.num_cols, A.nnz, A.val, A.row, A.col, 
  &(B->num_rows), &(B->num_cols), &(B->nnz), &(B->val), &(B->row), &(B->col) );
            B->memory_location = Magma_CPU;
            B->storage_type = Magma_CSR;
        }
        else{
            magma_z_sparse_matrix C, D;
            magma_z_mconvert( A, &C, A.storage_type, Magma_CSR);
            magma_z_mtranspose( C, &D );
            magma_z_mconvert( D, B, Magma_CSR, A.storage_type );
            magma_z_mfree(&C);
            magma_z_mfree(&D);
        }
    }
    else{
        magma_z_sparse_matrix C, D;
        magma_z_mtransfer( A, &C, A.memory_location, Magma_CPU);
        magma_z_mtranspose( C, &D );
        magma_z_mtransfer( D, B, Magma_CPU, A.memory_location );
        magma_z_mfree(&C);
        magma_z_mfree(&D);
    }

}

