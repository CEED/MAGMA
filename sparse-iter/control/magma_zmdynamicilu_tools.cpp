/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>



/**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place 
    where an element was deleted in the beginning of the loop. 
    In the new matrix, the added elements will then always be located at the 
    beginning of each row.
    
    More precisely,
    it inserts the new value in the value-pointer where the old element was
    located, changes the columnindex to the new index, modifies the row-pointer
    to point the this element, and sets the linked list element to the element
    where the row pointer pointed to previously.

    Arguments
    ---------

    @param[in]
    tri         magma_int_t
                info==0: lower trianguler, info==1: upper triangular.
                
    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    LU          magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *LU_new,
    magma_z_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    
    if(num_rm>=LU_new->nnz){
        //printf("num_rm = %d nnz:%d\n", num_rm, LU_new->nnz);
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmorderstatistics(
    LU_new->val, LU_new->col, LU_new->rowidx, LU_new->nnz, num_rm,  1, &element, queue ) );

    // insert the new elements
    // has to be sequential
    while( num_insert < num_rm ) {
        if(i>=LU_new->nnz){
            //printf("problem: exceeding candidate list: %d >%d\n", i, LU_new->nnz);
            break;
        }
        magma_int_t loc = rm_loc[ num_insert ];
        magma_index_t new_row = LU_new->rowidx[ i ]; 
        magma_index_t new_col = LU_new->col[ i ];
        
        magma_index_t old_rowstart = LU->row[ new_row ];
        if( new_col < LU->col[ old_rowstart ] ){
            //printf("insert at rowstart: (%d,%d)\n", new_row, new_col);
            LU->row[ new_row ] = loc;
            LU->list[ loc ] = old_rowstart;
            LU->rowidx[ loc ] = new_row;
            LU->col[ loc ] = new_col;
            LU->val[ loc ] = MAGMA_Z_ZERO;
            num_insert++;
        }
        else if( new_col == LU->col[ old_rowstart ] ){
            //printf("tried to insert duplicate!\n");
        }
        else{

            j = old_rowstart;
            jn = LU->list[j];
            // this will finish, as we consider the lower triangular
            // and we always have the diagonal!
            while( j!=0 ){
                if( LU->col[jn]==new_col ){
                    //printf("tried to insert duplicate!\n");
                    break;
                }else 
                if( LU->col[jn]>new_col ){
                    LU->list[j]=loc;
                    LU->list[loc]=jn;
                    LU->rowidx[ loc ] = new_row;
                    LU->col[ loc ] = new_col;
                    LU->val[ loc ] = MAGMA_Z_ZERO;
                    num_insert++;
                    break;
                }
                j=jn;
                jn=LU->list[jn];
            }
        }
        i++;
    }
cleanup:
    return info;
}


/**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold.

    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.
                
    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.
                
    @param[in]
    rowlock     omp_lock_t*
                OMP lock for the rows.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_rm_thrs(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    
    #pragma omp parallel for
    for( magma_int_t r=0;r<LU->num_rows;r++ ) {
        magma_int_t i = LU->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=LU->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( LU->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    LU->val[ i ] = MAGMA_Z_ZERO;
                    LU->list[ i ] = -1;
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    count_rm++;
                    omp_unset_lock(&(counter));
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( LU->row[r] == i ){
                            LU->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = LU->list[nexti];
                    }
                    else{
                        LU->list[lasti] = nexti;
                        i = nexti;
                        nexti = LU->list[nexti];
                    }
            }
            else{
                lasti = i;
                i = nexti;
                nexti = LU->list[nexti];
            }
            
        }
    }
    
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
    return info;
}


/**
    Purpose
    -------
    This routine computes the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicilu_set_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex element;
    magmaDoubleComplex *val;
    const magma_int_t ione = 1;
    
    CHECK( magma_zmalloc_cpu( &val, LU->nnz ));
    blasf77_zcopy(&LU->nnz, LU->val, &ione, val, &ione );
    
    // identify num_rm-th smallest element
    CHECK( magma_zorderstatistics(
    val, LU->nnz, num_rm, 0, &element, queue ) );
    //CHECK( magma_zsort( val, 0, LU->nnz, queue ) );
    //element = val[num_rm];
    *thrs = element;
    

cleanup:
    magma_free_cpu( val );
    return info;
}


/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    
    @param[in]
    A           magma_int_t
                System matrix A in CSR.

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation 
                The format is unsorted CSR, the list array is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_sweep(
    magma_z_matrix A,
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU->nnz; e++){
        if( LU->list[e]!=-1){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = LU->rowidx[ e ];
            magma_index_t col = LU->col[ e ];
            // as we look at the lower triangular, col<=row
            
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A.row[row]; i<A.row[row+1]; i++){
                if( A.col[i] == col ){
                    A_e = A.val[i];
                }
            }
            
            //now do the actual iteration
            i = LU->row[ row ]; 
            j = LU->row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = LU->col[i];
                jcol = LU->col[j];
                if( icol == jcol ){
                    lsum = LU->val[i] * LU->val[j];
                    sum = sum + lsum;
                    i = LU->list[i];
                    j = LU->list[j];
                }
                else if( icol<jcol ){
                    i = LU->list[i];
                }
                else {
                    j = LU->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;
            
            // write back to location e
            if ( row == col ){
                LU->val[ e ] = magma_zsqrt( A_e - sum );
            } else {
                LU->val[ e ] =  ( A_e - sum ) / LU->val[jold];
            }
        }// end check whether part of LU
        
    }// end omp parallel section
        
    return info;
}

 



/**
    Purpose
    -------
    This function computes the residuals for the candidates.

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                System matrix A.
    
    @param[in]
    LU          magma_z_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_residuals(
    magma_z_matrix A,
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU_new->nnz; e++){
        magma_int_t i,j,icol,jcol;
        
        magma_index_t row = LU_new->rowidx[ e ];
        magma_index_t col = LU_new->col[ e ];
        // as we look at the lower triangular, col<=row
        
        magmaDoubleComplex A_e = MAGMA_Z_ZERO;
        // check whether A contains element in this location
        for( i = A.row[row]; i<A.row[row+1]; i++){
            if( A.col[i] == col ){
                A_e = A.val[i];
            }
        }
        
        //now do the actual iteration
        i = LU.row[ row ]; 
        j = LU.row[ col ];
        magmaDoubleComplex sum = MAGMA_Z_ZERO;
        magmaDoubleComplex lsum = MAGMA_Z_ZERO;
        do{
            lsum = MAGMA_Z_ZERO;
            icol = LU.col[i];
            jcol = LU.col[j];
            if( icol == jcol ){
                lsum = LU.val[i] * LU.val[j];
                sum = sum + lsum;
                i = LU.list[i];
                j = LU.list[j];
            }
            else if( icol<jcol ){
                i = LU.list[i];
            }
            else {
                j = LU.list[j];
            }
        }while( i!=0 && j!=0 );
        
        // write back to location e
        LU_new->val[ e ] =  ( A_e - sum );
        
    }// end omp parallel section
        
    return info;
}




/**
    Purpose
    -------
    This function identifies the candidates.

    Arguments
    ---------
    
    @param[in]
    LU          magma_z_matrix
                Current LU approximation.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdynamicic_candidates(
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    LU_new->nnz = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    
    magma_index_t *numadd;
    CHECK( magma_index_malloc_cpu( &numadd, LU.num_rows+1 ));
    
    #pragma omp parallel for
    for( magma_int_t i = 0; i<LU.num_rows+1; i++ ){
        numadd[i] = 0;  
    }
     
 
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<LU.num_rows; row++){
        magma_index_t start = LU.row[ row ];
        
        magma_index_t lcol1 = start;
        // loop first element over row - only for elements smaller the diagonal
        while( LU.list[lcol1] != 0 ) {
            magma_index_t lcol2 = start;
            // loop second element over row
            while( lcol2 != lcol1 ) {
                // check whether the candidate already is included in LU
                magma_int_t exist = 0;
                magma_index_t col1 = LU.col[lcol1];
                magma_index_t col2 = LU.col[lcol2]; 
                // col1 is always larger as col2

                // we only look at the lower triangular part
                magma_index_t checkrow = col1;
                magma_index_t checkelement = col2;
                magma_index_t check = LU.row[ checkrow ];
                magma_index_t checkcol = LU.col[check];
                while( checkcol <= checkelement && check!=0 ) {
                    if( checkcol == checkelement ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    check = LU.list[ check ];
                    checkcol = LU.col[check];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                // go to next element
                lcol2 = LU.list[ lcol2 ];
            }
            // go to next element
            lcol1 = LU.list[ lcol1 ];
        }
    }
    
    // get the total candidate count
    LU_new->nnz = 0;
    // should become fan-in
    for( magma_int_t i = 0; i<LU.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }
    
    // allocate space
    CHECK( magma_zmalloc_cpu( &LU_new->val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &LU_new->rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &LU_new->col, LU_new->nnz ));
    LU_new->num_rows = LU.num_rows;
    LU_new->num_cols = LU.num_cols;
    
    // now insert - in parallel!
    #pragma omp parallel for
    for( magma_index_t row=0; row<LU.num_rows; row++){
        magma_index_t start = LU.row[ row ];
        magma_int_t ladd = 0;
        
        magma_index_t lcol1 = start;
        // loop first element over row
        while( LU.list[lcol1] != 0 ) {
            magma_index_t lcol2 = start;
            // loop second element over row
            while( lcol2 != lcol1 ) {
                // check whether the candidate already is included in LU
                magma_int_t exist = 0;
                
                magma_index_t col1 = LU.col[lcol1];
                magma_index_t col2 = LU.col[lcol2]; 
                // col1 is always larger as col2
 
                // we only look at the lower triangular part
                magma_index_t checkrow = col1;
                magma_index_t checkelement = col2;
                magma_index_t check = LU.row[ checkrow ];
                magma_index_t checkcol = LU.col[check];
                while( checkcol <= checkelement && check!=0 ) {
                    if( checkcol == checkelement ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    check = LU.list[ check ];
                    checkcol = LU.col[check];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allof for parallel insertion
                if( exist == 0 ){
                     //  printf("---------------->>>  candidate at (%d, %d)\n", col2, col1);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] = MAGMA_Z_ZERO; //or anything
                    LU_new->rowidx[ numadd[row] + ladd ] = col1;
                    LU_new->col[ numadd[row] + ladd ] = col2;
                    ladd++;
                }
                // go to next element
                lcol2 = LU.list[ lcol2 ];
            }
            // go to next element
            lcol1 = LU.list[ lcol1 ];
        }
    }

cleanup:
    magma_free_cpu( numadd );
    omp_destroy_lock(&(counter));
    return info;
}


#endif

