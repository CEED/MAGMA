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


using namespace std;




/**
    Purpose
    -------

    Passes a CSR matrix to MAGMA.

    Arguments
    ---------

    @param[in]
    m           magma_int_t 
                number of rows

    @param[in]
    n           magma_int_t 
                number of columns

    @param[in]
    row         magma_index_t*
                row pointer

    @param[in]
    col         magma_index_t*
                column indices

    @param[in]
    val         magmaDoubleComplex*
                array containing matrix entries

    @param[out]
    A           magma_z_sparse_matrix*
                matrix in magma sparse matrix format
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zcsrset(
    magma_int_t m, 
    magma_int_t n, 
    magma_index_t *row, 
    magma_index_t *col,     
    magmaDoubleComplex *val,
    magma_z_sparse_matrix *A,
    magma_queue_t queue )
{
    A->num_rows = m;
    A->num_cols = n;
    A->nnz = row[m];
    A->storage_type = Magma_CSR;
    A->memory_location = Magma_CPU;
    A->val = val;
    A->col = col;
    A->row = row;
    A->fill_mode = Magma_FULL;

    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------

    Passes a MAGMA matrix to CSR structure.

    Arguments
    ---------

    @param[in]
    A           magma_z_sparse_matrix
                magma sparse matrix in CSR format

    @param[out]
    m           magma_int_t 
                number of rows

    @param[out]
    n           magma_int_t 
                number of columns

    @param[out]
    row         magma_index_t*
                row pointer

    @param[out]
    col         magma_index_t*
                column indices

    @param[out]
    val         magmaDoubleComplex*
                array containing matrix entries

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zcsrget(
    magma_z_sparse_matrix A,
    magma_int_t *m, 
    magma_int_t *n, 
    magma_index_t **row, 
    magma_index_t **col, 
    magmaDoubleComplex **val,
    magma_queue_t queue )
{
    if ( A.memory_location == Magma_CPU && A.storage_type == Magma_CSR ) {

        *m = A.num_rows;
        *n = A.num_cols;
        *val = A.val;
        *col = A.col;
        *row = A.row;
    } else {
        magma_z_sparse_matrix A_CPU, A_CSR;
        magma_zmtransfer( A, &A_CPU, A.memory_location, Magma_CPU, queue ); 
        magma_zmconvert( A_CPU, &A_CSR, A_CPU.storage_type, Magma_CSR, queue ); 
        magma_zcsrget( A_CSR, m, n, row, col, val, queue );
        magma_zmfree( &A_CSR, queue );
        magma_zmfree( &A_CPU, queue );
    }
    return MAGMA_SUCCESS;
}


