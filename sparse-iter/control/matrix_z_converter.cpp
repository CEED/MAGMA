/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <math.h>
#include "../include/magmasparse_z.h"

magma_int_t z_array2csr( magma_int_t *m, magma_int_t *n, magma_int_t *nnz, magmaDoubleComplex*a, magmaDoubleComplex **val, magma_int_t **row, magma_int_t **col )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Converts a matrix stored in array format into a CSR storage format

    Arguments
    =========

    magma_int_t *m                       number of rows in matrix 
    magma_int_t *n                       number of columns in matrix
    magma_int_t *nnz                     number of nonzeros (is computed) 
    magmaDoubleComplex*a                 input matrix in array format
    magmaDoubleComplex **val             value array of CSR output     
    magma_int_t **row                    row pointer of CSR output
    magma_int_t **col                    column indices of CSR output

    =====================================================================  */
  *nnz=0;
  for( magma_int_t i=0; i<(*n)*(*n); i++ ){
    if( MAGMA_Z_REAL(a[i])!=0.0 )
      (*nnz)++;
  }

  (*val) = new magmaDoubleComplex[*nnz];
  (*col) = ( magma_int_t* )malloc((*nnz)*sizeof( magma_int_t ));
  (*row) = ( magma_int_t* )malloc((*m+1)*sizeof( magma_int_t ));

  magma_int_t i = 0;
  magma_int_t j = 0;
  magma_int_t k = 0;

  for(i=0; i<(*n)*(*m); i++)
  {
    if( i%(*n)==0 )
  {
    (*row)[k] = j;
    k++;
  }
    if( MAGMA_Z_REAL(a[i])!=0 )
    {
      (*val)[j] = a[i];
      (*col)[j] = i%(*n);
      j++;
    }

  }
  (*row)[*n]=*nnz;
}


magma_int_t z_csr2array( magma_int_t *m, magma_int_t *n, magma_int_t *nnz, magmaDoubleComplex *val, magma_int_t *row, magma_int_t *col, magmaDoubleComplex*b )
{
/*  -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    =======

    Converts a matrix stored in array format into a CSR storage format

    Arguments
    =========

    magma_int_t *m                       number of rows in matrix 
    magma_int_t *n                       number of columns in matrix
    magma_int_t *nnz                     number of nonzeros of input 
    magmaDoubleComplex **val             value array of CSR input     
    magma_int_t **row                    row pointer of CSR input
    magma_int_t **col                    column indices of CSR input
    magmaDoubleComplex*b                 output matrix in array format

    =====================================================================  */
  magma_int_t i=0,j=0;
  magma_int_t nn = (*n)*(*m);

  for( i=0; i<nn; i++ )
    (b)[i] = MAGMA_Z_MAKE( 0.0, 0.0 );


  for( i=0; i<*n; i++ )
  {
    magma_int_t rowtemp1 = row[i];
    magma_int_t rowtemp2 = row[i+1];
    for( j=0; j<rowtemp2-rowtemp1; j++ )  
      (b)[i*(*n)+col[rowtemp1+j]] = val[rowtemp1+j];
  }
}
