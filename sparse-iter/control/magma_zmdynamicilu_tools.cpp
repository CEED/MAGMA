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
    Screens the new candidates for duplicates and sets the value for those 
    elements to zero. In the following sort-step, the elements are tehn filtered
    out. This idea is according to Enriques comment that the thread synchronization
    is much more expensive than doing another scan step.
    
    Additionally, the list of components is then sorted 
    (largest elements in the beginning).
    
    Finally, a second sweep over all elements, it is indicated whether an element
    is alone/first in this row, or whether there are other elements in this row.
    

    Arguments
    ---------
    
    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmeliminate_duplicates(
    magma_int_t num_rm,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1, chunksize;
    magmaDoubleComplex element;
    magma_int_t i, count = 0;
    omp_lock_t counter;
    omp_init_lock(&(counter));
    magma_int_t duplicate = 0;
    double thrs = 1.6;
    magma_int_t approx, rm_approx, success = 0;
    const magma_int_t ione = 1;
    const magma_int_t iten = 1;

    
    
    //.// Idea: instead of order-statistics, identify 120th largest element, then sweep over, and copy only if not yet included.
    //.// finally, order statistics on remaining elements
    while( success !=1 ){
       
       info = magma_zparilut_set_approx_thrs( (int)num_rm*thrs, LU_new, 1, &element, queue );
        //printf("threshold:%f LU_new.nnz:%d\n", element, LU_new->nnz);
        //CHECK( magma_zorderstatistics(
        //LU_new->val, LU_new->nnz, (int)(num_rm*thrs),  1, &element, queue ) );
        //printf("done orderstatistics\n");
    
        count = 0;
        #pragma omp parallel for
        for( magma_int_t i=0; i<LU_new->nnz; i++ ){
            magma_int_t localcount;
            if( MAGMA_Z_ABS(LU_new->val[i]) >= MAGMA_Z_ABS(element) ){
                omp_set_lock(&(counter));
                duplicate = 0;
                magma_int_t col = LU_new->col[i];
                magma_int_t row = LU_new->rowidx[i];
                #pragma omp parallel for
                for( magma_int_t j=0;j<count;j++ ){
                    if( LU_new->rowidx[j] == row ){
                        if( LU_new->col[j] == col ){
                            // two times the same candidate
                            duplicate = 1;
                            j=count;
                        }
                    }
                }
                if( duplicate == 0 ){
                    
                    
                    //omp_set_lock(&(counter));
                    localcount = count;
                    count=count+1;
                    //omp_unset_lock(&(counter));
                    LU_new->col[localcount] = col;
                    LU_new->rowidx[localcount] = row; 
                    magmaDoubleComplex valt = LU_new->val[i];
                    LU_new->val[i] = MAGMA_Z_ZERO;
                    LU_new->val[localcount] = valt;
                    
                } else {
                    LU_new->val[i] = MAGMA_Z_ZERO;
                    //printf("found duplicate!\n");
                }
                omp_unset_lock(&(counter));
            }
        }

        
        if( count < num_rm ){
             printf("%% error: %d < %d -> increase threshold\n", count, num_rm);
            // make the threshold smaller:
            element = element * MAGMA_Z_MAKE(.01,0.0);
            #pragma omp parallel for
            for( magma_int_t i=count; i<LU_new->nnz; i++ ){
                magma_int_t localcount;
                if( MAGMA_Z_ABS(LU_new->val[i]) >= MAGMA_Z_ABS(element) ){
                    omp_set_lock(&(counter));
                    duplicate = 0;
                    magma_int_t col = LU_new->col[i];
                    magma_int_t row = LU_new->rowidx[i];
                    #pragma omp parallel for
                    for( magma_int_t j=0;j<count;j++ ){
                        if( LU_new->rowidx[j] == row ){
                            if( LU_new->col[j] == col ){
                                // two times the same candidate
                                duplicate = 1;
                                j=count;
                            }
                        }
                    }
                    if( duplicate == 0 ){
                        
                        
                        //omp_set_lock(&(counter));
                        localcount = count;
                        count=count+1;
                        //omp_unset_lock(&(counter));
                        LU_new->col[localcount] = col;
                        LU_new->rowidx[localcount] = row; 
                        magmaDoubleComplex valt = LU_new->val[i];
                        LU_new->val[i] = MAGMA_Z_ZERO;
                        LU_new->val[localcount] = MAGMA_Z_ZERO;
                        LU_new->val[localcount] = valt;
                        if( count > num_rm+5 ){
                            i=LU_new->nnz;    
                        }
                        
                    } else {
                        LU_new->val[i] = MAGMA_Z_ZERO;
                        //printf("found duplicate!\n");
                    }
                    omp_unset_lock(&(counter));
                }
            }
            
            success=1;
             printf("%% now: %d > %d\n", count, num_rm);
          //exit(-1);
        //} else if ( count > num_rm*thrs ) {
        //  printf("%% error: something wrong.\n", count, num_rm); fflush(stdout);
        } else if(count > num_rm || count <= (int)num_rm*1.2 ){      
            //printf("order again %d %d\n", count, num_rm);
            //CHECK( magma_zmorderstatistics(
            //    LU_new->val, LU_new->col, LU_new->rowidx, count, num_rm,  1, &element, queue ) );
            success=1;
        } else if( count > (int)num_rm*1.2 ){   
            //thrs = thrs-0.1;
            //printf("%% error: %d > %d -> decrease threshold\n", count, (int)num_rm*1.2);
            success=1;
        }else{
            success=1;
        }
           
    }
//printf("%%done:%d %d %d .\n", LU_new->nnz, count, num_rm); fflush(stdout);
    // as we have no longer duplicates, we just need to sort num_rm elements to the front
    
     //      printf("unsorted:\n");
     //  for(int z=1;z<LU_new->nnz;z++)
     //      printf("(%d,%d)=%f\t", LU_new->rowidx[z],LU_new->col[z],LU_new->val[z]);
     //   printf("\n");
    
    //################
    
   
   
   //rintf("%%done2.\n"); fflush(stdout);
   //           printf("sorted:\n");
   //   for(int z=1;z<LU_new->nnz;z++)
   //       printf("(%d,%d)=%f\t", LU_new->rowidx[z],LU_new->col[z],LU_new->val[z]);
   //    printf("\n");
    // Now another sweep. 
    // We check whether an element is the only one in a row, or whether there are more
    //#pragma omp parallel for
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_rm; i++ ) {
        magma_int_t row = LU_new->rowidx[i];
        magma_int_t locallist = 0;
        // check whether there are more in this row
        for( magma_int_t j=i+1; j<num_rm; j++ ) { // only elements right
            if( LU_new->rowidx[j] == row ){
                locallist++;
                j=num_rm;
            }
        }
        // check whether it is not the first in this row
        for( magma_int_t j=0; j<i; j++ ) { // only elements right
            if( LU_new->rowidx[j] == row ){
                locallist = -2;
                j=i;
            }
        }
        LU_new->list[i] = locallist;
    }
  //  printf("%%done3.\n"); fflush(stdout);
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_rm; i++ ) {
        magma_int_t col = LU_new->col[i];
        magma_int_t locallist = 0;
        // check whether there are more in this row
        for( magma_int_t j=i+1; j<num_rm; j++ ) { // only elements right
            if( LU_new->col[j] == col ){
                locallist++;
                j=num_rm;
            }
        }
        // check whether it is not the first in this row
        for( magma_int_t j=0; j<i; j++ ) { // only elements right
            if( LU_new->col[j] == col ){
                locallist = -2;
                j=i;
            }
        }
        LU_new->row[i] = locallist;
    }

    
   // printf("%%done4.\n"); fflush(stdout);
cleanup:
    omp_destroy_lock(&(counter));
    return info;
}




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
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    rm_loc      magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    rm_loc2     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    U           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_insert_LU(
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_index_t *rm_loc2,
    magma_z_matrix *LU_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t duplL = 0;
    magma_int_t duplU = 0;
    
    // first part L
    #pragma omp parallel for
    for (magma_int_t r=0; r<num_rm; r++){
        // case only element in row
        if( LU_new->list[r] == 0 ){
            magma_int_t loc = rm_loc[ r ];
            magma_index_t new_row = LU_new->rowidx[ r ]; 
            magma_index_t new_col = LU_new->col[ r ];
            magma_index_t old_rowstart = L->row[ new_row ];
            
            if( new_col < L->col[ old_rowstart ] ){
                //printf("%insert in L: (%d,%d)\n", new_row, new_col);
                L->row[ new_row ] = loc;
                L->list[ loc ] = old_rowstart;
                L->rowidx[ loc ] = new_row;
                L->col[ loc ] = new_col;
                L->val[ loc ] = MAGMA_Z_ZERO;
            }
            else if( new_col == L->col[ old_rowstart ] ){
                printf("%% tried to insert duplicate! case 1\n");
            }
            else{
                magma_int_t j = old_rowstart;
                magma_int_t jn = L->list[j];
                // this will finish, as we consider the lower triangular
                // and we always have the diagonal!
                while( j!=0 ){
                    if( L->col[jn]==new_col ){
                                //rm_loc[duplL] = loc;
                                //duplL++;
                                // L->list[j]=loc;
                                // L->list[loc]=jn;
                                // L->rowidx[ loc ] = new_row;
                                // L->col[ loc ] = new_col-1;
                                // LU_new->col[ r ] = new_col-1;
                                // L->val[ loc ] =  MAGMA_Z_ZERO;                   
                        printf("%% tried to insert duplicate case 1 2 in L: (%d %d) \n", new_row, new_col);
                                 // if( U->col[ j ] == new_col-1){
                                 //     printf("%%problem in L: (%d,%d)\n", new_row, new_col);
                                 // }
            //for(int z=0;z<num_rm;z++){
            //    if(LU_new->col[z]==new_col && LU_new->rowidx[z]==new_row){
            //    printf("here:%d \n\n",z);
            //    }
            //}

                        j=0; //break;
                    }else if( L->col[jn]>new_col ){

                        //printf("%insert in L: (%d,%d)\n", new_row, new_col);
                        L->list[j]=loc;
                        L->list[loc]=jn;
                        L->rowidx[ loc ] = new_row;
                        L->col[ loc ] = new_col;
                        L->val[ loc ] = MAGMA_Z_ZERO;
                        j=0; //break;
                        
                    } else{
                        j=jn;
                        jn=L->list[jn];
                    }
                }
            }
        }
        // case multiple elements in row
        else if( LU_new->list[r] > 0 ){
            magma_index_t new_row = LU_new->rowidx[ r ]; 
            
            for( magma_int_t r2 = r; r2<num_rm; r2++ ){
                if( LU_new->rowidx[r2] == new_row ){
                    magma_int_t loc = rm_loc[ r2 ];
                    magma_index_t new_col = LU_new->col[ r2 ];
                    magma_index_t old_rowstart = L->row[ new_row ];
                    if( new_col < L->col[ old_rowstart ] ){
                        //printf("%insert in L: (%d,%d)\n", new_row, new_col);
                        L->row[ new_row ] = loc;
                        L->list[ loc ] = old_rowstart;
                        L->rowidx[ loc ] = new_row;
                        L->col[ loc ] = new_col;
                        L->val[ loc ] = MAGMA_Z_ZERO;
                    }
                    else if( new_col == L->col[ old_rowstart ] ){
                         printf("%%tried to insert duplicate case 1 !\n");
                    }
                    else{
                        magma_int_t j = old_rowstart;
                        magma_int_t jn = L->list[j];
                        // this will finish, as we consider the lower triangular
                        // and we always have the diagonal!
                        while( j!=0 ){
                            if( L->col[jn]==new_col ){
                        printf("%% tried to insert duplicate case 2 2 in L: (%d %d) \n", new_row, new_col);
                             // if( L->col[ j ] == new_col-1){
                             //         printf("%%problem in L: (%d,%d)\n", new_row, new_col);
                             // }
                                // L->list[j]=loc;
                                // L->list[loc]=jn;
                                // L->rowidx[ loc ] = new_row;
                                // L->col[ loc ] = new_col-1;
                                // LU_new->col[ r2 ] = new_col-1;
                                // L->val[ loc ] = MAGMA_Z_ZERO;
                                //rm_loc[duplL] = loc;
                                //duplL++;


                for(int z=0;z<num_rm;z++){
                                if(LU_new->col[z]==new_col && LU_new->rowidx[z]==new_row){
                                printf("here:%d \n\n",z);
                                }
                        }           
                                j=0; //break;
                            }else if( L->col[jn]>new_col ){
                                //printf("%insert in L: (%d,%d)\n", new_row, new_col);
                                L->list[j]=loc;
                                L->list[loc]=jn;
                                L->rowidx[ loc ] = new_row;
                                L->col[ loc ] = new_col;
                                L->val[ loc ] = MAGMA_Z_ZERO;
                                j=0; //break;
                                
                            } else{
                                j=jn;
                                jn=L->list[jn];
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    
    // second part U
    #pragma omp parallel for
    for (magma_int_t r=0; r<num_rm; r++){
        // case only element in row
        if( LU_new->row[r] == 0 ){
            magma_int_t loc = rm_loc2[ r ];
            magma_index_t new_row = LU_new->col[ r ]; 
            magma_index_t new_col = LU_new->rowidx[ r ];
            magma_index_t old_rowstart = U->row[ new_row ];
            magma_int_t jnT = old_rowstart;
            magma_int_t jT;
            // diagonal element always exists!
            while( jnT!=0 ){
                jT=jnT;
                jnT=U->list[jnT];
                if( jnT == 0 ){
                    //printf("%%insert in U: %d (%d,%d)\n", r, new_row, new_col);
                    U->list[jT]=loc;
                    U->list[loc]=jnT;
                    U->rowidx[ loc ] = new_row;
                    U->col[ loc ] = new_col;
                    U->val[ loc ] = MAGMA_Z_ONE;
                    jnT=0; //break;
                }else if( U->col[jnT]==new_col ){
                            printf("%%insert duplicate in U 1 2 : %d (%d,%d)\n", r, new_row, new_col);
                                 // if( U->col[ jT ] == new_col-1){
                                 //     printf("%%problem in U: (%d,%d)\n", new_row, new_col);
                                 // }
                                //rm_loc2[duplU] = loc;
                                //duplU++;
                                // U->list[jT]=loc;
                                // U->list[loc]=jnT;
                                // U->rowidx[ loc ] = new_row;
                                // U->col[ loc ] = new_col-1;
                                // U->val[ loc ] = MAGMA_Z_ZERO;
            jnT=0; //break;
                }else if( U->col[jnT]>new_col ){
                    //printf("%%insert in U: %d (%d,%d)->(%d,%d)->(%d,%d) in location rm[%d] = %d\n", r, U->rowidx[jT], U->col[jT], new_row, new_col, U->rowidx[jnT], U->col[jnT], r, loc);
                    U->list[jT]=loc;
                    U->list[loc]=jnT;
                    U->rowidx[ loc ] = new_row;
                    U->col[ loc ] = new_col;
                    U->val[ loc ] = MAGMA_Z_ONE;
                    jnT=0; //break;
                    
                } 
            }
        }
        // case multiple elements in row
        else if( LU_new->row[r] > 0 ){
            magma_index_t new_row = LU_new->col[ r ]; 
            
            for( magma_int_t r2 = r; r2<num_rm; r2++ ){
                if( LU_new->col[r2] == new_row ){
                    magma_int_t loc = rm_loc2[ r2 ];
                    magma_index_t new_row = LU_new->col[ r2 ]; 
                    magma_index_t new_col = LU_new->rowidx[ r2 ];
                    magma_index_t old_rowstart = U->row[ new_row ];
                    magma_int_t jnT = old_rowstart;
                    magma_int_t jT;
                    // diagonal element always exists!
                    while( jnT!=0 ){
                        jT=jnT;
                        jnT=U->list[jnT];
                        if( jnT == 0 ){
                            //printf("%%insert in U: (%d,%d)\n", new_row, new_col);
                            U->list[jT]=loc;
                            U->list[loc]=jnT;
                            U->rowidx[ loc ] = new_row;
                            U->col[ loc ] = new_col;
                            U->val[ loc ] = MAGMA_Z_ONE;
                            jnT=0; //break;
                        }else if( U->col[jnT]==new_col ){
                              printf("%%insert duplicate in U 2 2: %d %d (%d,%d)\n", r, r2, new_row, new_col);
                                //rm_loc2[duplU] = loc;
                                 //duplU++;
                                 // if( U->col[ jT ] == new_col-1){
                                 //     printf("%%problem in U: (%d,%d)\n", new_row, new_col);
                                 // }
                             //   U->list[jT]=loc;
                             // U->list[loc]=jnT;
                             // U->rowidx[ loc ] = new_row;
                             // U->col[ loc ] = new_col-1;
                             // U->val[ loc ] = MAGMA_Z_ZERO;                            
                                jnT=0; //break;
                        }else if( U->col[jnT]>new_col ){
                            //printf("%%insert in U case 2: %d %d (%d,%d)->(%d,%d)->(%d,%d)\n", r, r2, U->rowidx[jT], U->col[jT], new_row, new_col, U->rowidx[jnT], U->col[jnT]);
                            U->list[jT]=loc;
                            U->list[loc]=jnT;
                            U->rowidx[ loc ] = new_row;
                            U->col[ loc ] = new_col;
                            U->val[ loc ] = MAGMA_Z_ONE;
                            jnT=0; //break;
                            
                        } 
                    }
                }
            }
        }
    }
    //LU_new->diameter = duplL;
    //LU_new->blocksize = duplU;
    
cleanup:
     return info;
}




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
    rm_loc2     magma_index_t*
                List containing the locations of the deleted elements.
                
    @param[in]
    LU_new      magma_z_matrix
                Elements that will be inserted stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted. 
                The format is unsorted CSR, list is used as linked 
                list pointing to the respectively next entry.
                
    @param[in,out]
    U           magma_z_matrix*
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
magma_zparict_insert_LU(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_index_t *rm_loc2,
    magma_z_matrix *LU_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn, jT, jnT;
    
    magma_int_t i=0, iT=0;
    magma_int_t num_insert = 0, num_insertT;
    int loc_i=0, loc_iT=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    magma_int_t secondthread = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    if( num_threads>1 ){
        secondthread = 1;
    }

    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }

    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));

    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }

   // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    #pragma omp parallel
    {
        magma_int_t tid = omp_get_thread_num();
    if(tid == 0 ){
    //{
    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        //magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                //#pragma omp critical(num_insert)
                {
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = rowidx[ loc_i ]; 
                

                
                //#pragma omp critical(rowlock__)
                //{
                //    omp_set_lock( &(rowlock[new_row]) );
                //}
                magma_index_t new_col = col[ loc_i ];
                magma_index_t old_rowstart = L->row[ new_row ];
                
                if( new_col < L->col[ old_rowstart ] ){
                 //   printf("%insert in L: (%d,%d)\n", new_row, new_col);
                    L->row[ new_row ] = loc;
                    L->list[ loc ] = old_rowstart;
                    L->rowidx[ loc ] = new_row;
                    L->col[ loc ] = new_col;
                    L->val[ loc ] = MAGMA_Z_ZERO;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == L->col[ old_rowstart ] ){
                    printf("tried to insert duplicate!\n");
                }
                else{
        
                    j = old_rowstart;
                    jn = L->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    while( j!=0 ){
                        if( L->col[jn]==new_col ){
                            printf("tried to insert duplicate!\n");
                            j=0; //break;
                        }else if( L->col[jn]>new_col ){
                            L->list[j]=loc;
                            L->list[loc]=jn;
                            L->rowidx[ loc ] = new_row;
                            L->col[ loc ] = new_col;
                            L->val[ loc ] = MAGMA_Z_ZERO;
                            success[ tid*8 ] = 1;
                            j=0; //break;
                            
                        } else{
                            j=jn;
                            jn=L->list[jn];
                        }
                    }
                }
                //#pragma omp critical(rowlock__)
                //omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
        //printf("L done\n");
        success[tid*8]= ( tid<num_rm ) ? 1 : -1;
        insert_loc[tid*8] = -1;
    }
    
    

    // Part for U
    /*
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    */
    
    if( tid == secondthread ){
    //{
    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_iT=0; loc_iT<LU_new->nnz; loc_iT++ ) {

        //magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                //#pragma omp critical(num_insertT)
                {
                    insert_loc[ tid*8 ] = num_insertT;
                    num_insertT++;
                }
                success[ tid*8 ] = 0;

            }

            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }

            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc2[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = col[ loc_iT ]; 
                

                
                //#pragma omp critical(rowlock__)
                //{
                //    omp_set_lock( &(rowlock[new_row]) );
                //}
                magma_index_t new_col = rowidx[ loc_iT ];
                magma_index_t old_rowstart = U->row[ new_row ];
                jnT = old_rowstart;
                // diagonal element always exists!
                while( jnT!=0 ){
                    jT=jnT;
                    jnT=U->list[jnT];
                    if( jnT == 0 ){
                        U->list[jT]=loc;
                        U->list[loc]=jnT;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ONE;
                        success[ tid*8 ] = 1;
                        jnT=0; //break;
                    }else if( U->col[jnT]==new_col ){
                        jnT=0; //break;
                    }else if( U->col[jnT]>new_col ){
                        U->list[jT]=loc;
                        U->list[loc]=jnT;
                        U->rowidx[ loc ] = new_row;
                        U->col[ loc ] = new_col;
                        U->val[ loc ] = MAGMA_Z_ONE;
                        success[ tid*8 ] = 1;
                        jnT=0; //break;
                        
                    } 
                }
                //#pragma omp critical(rowlock__)
                //omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    //printf("U done\n");
    }
    
    }// end parallel
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
    return info;
}


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
magma_zparict_insert(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *LU_new,
    magma_z_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    int loc_i=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    
    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));
    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }
    // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = rowidx[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = col[ loc_i ];
                magma_index_t old_rowstart = LU->row[ new_row ];
                
                           //                         printf("tid*8:%d loc_i:%d loc_num_insert:%d num_rm:%d target loc:%d  element (%d,%d)\n",
                           //     tid*8, loc_i, insert_loc[ tid*8 ], num_rm, loc, new_row, new_col); fflush(stdout);
                           //     printf("-->(%d,%d)\n", new_row, new_col); fflush(stdout);

                

                if( new_col < LU->col[ old_rowstart ] ){
                    LU->row[ new_row ] = loc;
                    LU->list[ loc ] = old_rowstart;
                    LU->rowidx[ loc ] = new_row;
                    LU->col[ loc ] = new_col;
                    LU->val[ loc ] = MAGMA_Z_ZERO;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == LU->col[ old_rowstart ] ){
                    ; //printf("tried to insert duplicate!\n");
                }
                else{
        
                    j = old_rowstart;
                    jn = LU->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    while( j!=0 ){
                        if( LU->col[jn]==new_col ){
                            //printf("tried to insert duplicate!\n");
                            j=0; //break;
                        }else if( LU->col[jn]>new_col ){
                            LU->list[j]=loc;
                            LU->list[loc]=jn;
                            LU->rowidx[ loc ] = new_row;
                            LU->col[ loc ] = new_col;
                            LU->val[ loc ] = MAGMA_Z_ZERO;
                            success[ tid*8 ] = 1;
                            j=0; //break;
                            
                        } else{
                            j=jn;
                            jn=LU->list[jn];
                        }
                    }
                }
                //#pragma omp critical(rowlock__)
                //{
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
    return info;
}


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
magma_zparict_insert_U(
    magma_int_t tri,
    magma_int_t num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *LU_new,
    magma_z_matrix *LU,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *val;
    magma_index_t *col;
    magma_index_t *rowidx;
        
    magmaDoubleComplex element;
    magma_int_t j,jn;
    
    magma_int_t i=0;
    magma_int_t num_insert = 0;
    int loc_i=0;
    int abort = 0;
    magma_int_t *success;
    magma_int_t *insert_loc;
    magma_int_t num_threads = 0;
    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }
    
    CHECK( magma_imalloc_cpu( &success, num_threads*8 ));
    CHECK( magma_imalloc_cpu( &insert_loc, num_threads*8 ));
    //omp_lock_t rowlock[LU->num_rows];
    #pragma omp parallel for
    for (magma_int_t r=0; r<omp_get_num_threads(); r++){
        success[r*8]= ( r<num_rm ) ? 1 : -1;
        insert_loc[r*8] = -1;
    }
    if(num_rm>=LU_new->nnz){
        printf("error: try to remove too many elements\n.");
        goto cleanup;
    }
    // identify num_rm-th largest elements and bring them to the front
    CHECK( magma_zmalloc_cpu( &val, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU_new->nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU_new->nnz ));
    #pragma omp parallel for
    for( magma_int_t r=0; r<LU_new->nnz; r++ ) {
        col[r] = LU_new->col[r];
        rowidx[r] = LU_new->rowidx[r];
        val[r] = LU_new->val[r];
    }
    // this is usually sufficient to have the large elements in front
    CHECK( magma_zmorderstatistics(
    val, col, rowidx, LU_new->nnz, num_rm + (LU_new->nnz-num_rm)*.1,  1, &element, queue ) );

    CHECK( magma_zmorderstatistics(
    val, col, rowidx, num_rm + (LU_new->nnz-num_rm)*.1, num_rm, 1, &element, queue ) );
    // insert the new elements
    // has to be sequential

    //#pragma omp parallel for private(loc_i) schedule(static,3) shared(num_insert)
    for(int loc_i=0; loc_i<LU_new->nnz; loc_i++ ) {
        magma_int_t tid = omp_get_thread_num();
        if( success[ tid*8 ] > -1 ){
            if( success[ tid*8 ] == 1 ){
                #pragma omp critical(num_insert)
                {
                    printf("->next element->");
                    insert_loc[ tid*8 ] = num_insert;
                    num_insert++;
                }
                success[ tid*8 ] = 0;
            }
            if( insert_loc[ tid*8 ] >= num_rm ){
                // enough elements added
                success[ tid*8 ] = -1;
            }
            if( success[ tid*8 ] > -1 ){
                magma_int_t loc = rm_loc[ insert_loc[ tid*8 ] ];
                magma_index_t new_row = col[ loc_i ]; 
                

                
                #pragma omp critical(rowlock__)
                {
                    omp_set_lock( &(rowlock[new_row]) );
                }
                magma_index_t new_col = rowidx[ loc_i ];
                magma_index_t old_rowstart = LU->row[ new_row ];
                
                           //                         printf("tid*8:%d loc_i:%d loc_num_insert:%d num_rm:%d target loc:%d  element (%d,%d)\n",
                           //     tid*8, loc_i, insert_loc[ tid*8 ], num_rm, loc, new_row, new_col); fflush(stdout);
                                //printf("-->(%d,%d)\t", new_row, new_col); fflush(stdout);

                

                if( new_col < LU->col[ old_rowstart ] ){
                    LU->row[ new_row ] = loc;
                    LU->list[ loc ] = old_rowstart;
                    LU->rowidx[ loc ] = new_row;
                    LU->col[ loc ] = new_col;
                    LU->val[ loc ] = MAGMA_Z_ONE;
                    success[ tid*8 ] = 1;
                }
                else if( new_col == LU->col[ old_rowstart ] ){
                    printf("tried to insert duplicate!\n"); fflush(stdout);
                }
                else{
        
                    j = old_rowstart;
                    jn = LU->list[j];
                    // this will finish, as we consider the lower triangular
                    // and we always have the diagonal!
                    magma_int_t breakpoint = 0;
                    //if ( new_row == 0 ) {
                    //    LU->row[new_row+1];
                    //}
                    while( j!=breakpoint ){
                        if( LU->col[jn]==new_col ){
                            printf("tried to insert duplicate!\n");
                            j=breakpoint; //break;
                        }else if( LU->col[jn]>new_col ){
                            
                            
                            // printf("%insert: (%d,%d)\t", new_row, new_col); fflush(stdout);
                            LU->list[j]=loc;
                            LU->list[loc]=jn;
                            LU->rowidx[ loc ] = new_row;
                            LU->col[ loc ] = new_col;
                            LU->val[ loc ] = MAGMA_Z_ONE;
                            success[ tid*8 ] = 1;
                            j=breakpoint; //break;
                            
                        } else{
                            j=jn;
                            jn=LU->list[jn];
                        }
                    }//printf("done\n"); fflush(stdout);
                }
                //#pragma omp critical(rowlock__)
                //{
                omp_unset_lock( &(rowlock[new_row]) );
            }
        }
    }// abort
    
cleanup:
    magma_free_cpu( success );
    magma_free_cpu( insert_loc );
    
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( rowidx );
    
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
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
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
magma_zparilut_rm_thrs(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    #pragma omp parallel for
    for( magma_int_t r=0; r < LU->num_rows; r++ ) {
        magma_int_t i = LU->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=LU->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( LU->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    magmaDoubleComplex valt = LU->val[ i ];
                    LU->val[ i ] = MAGMA_Z_ZERO;
                    LU->list[ i ] = -1;
                    if( LU->col[ i ] == r ){
                        printf("error: try to rm diagonal in L.\n");   
                    }
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                    LU_new->col[ count_rm+offset ] = LU->col[ i ];
                    LU_new->rowidx[ count_rm+offset ] = r;
                    LU_new->val[ count_rm+offset ] = valt; // MAGMA_Z_MAKE(1e-14,0.0);
                    // printf(" in L rm: (%d,%d) [%d] \n", r, LU->col[ i ], count_rm); fflush(stdout);
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
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    omp_destroy_lock(&(counter));
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
    U          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
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
magma_zparilut_rm_thrs_U(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *U,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    //omp_lock_t counter;
    //omp_init_lock(&(counter));
    
     //comment out for now
    
    // set elements to zero
    #pragma omp parallel for
    for( magma_int_t rm=0; rm < *num_rm; rm++ ) {
        magma_int_t rm_row = LU_new->col[rm];
        magma_int_t rm_col = LU_new->rowidx[rm];
        magma_int_t elo = U->row[rm_row];
        magma_int_t el = U->list[elo];
        magma_int_t success = 0;
       //printf("rm: (%d,%d) in", rm_row, rm_col); fflush(stdout);
      //  printf(" el:%d\t", el);                    fflush(stdout);
        do{
            if( U->col[elo] == rm_col ){
                U->val[elo] = MAGMA_Z_ZERO;
                rm_loc[ rm ] = elo;
                el = 0;
                 //printf("rm tid %d location [%d]=%d (%d %d) done.\n", rm, rm,el, rm_row, rm_col); fflush(stdout);
                    
            }
            else if( U->col[elo] > rm_col ){
                //U->val[elo] = MAGMA_Z_ZERO;
                //success = 1;
                printf("%% error: does not exist in U: %d (%d,%d)  %d (%d,%d) -> %d (%d,%d) row starts with (%d,%d) \n", rm, rm_row, rm_col, elo, U->rowidx[elo],U->col[elo],el,U->rowidx[el],U->col[el], U->rowidx[U->row[rm_row]], U->col[U->row[rm_row]]);fflush(stdout);
                el = 0;
            }
            elo = el;
            el = U->list[el];
            //printf("Ucol:%d->%d  ", el, U->col[el]);
        }while( elo != 0 );
    }
    //printf("first part done :%d elements.\n", *num_rm); fflush(stdout);
  
        
    #pragma omp parallel for
    for( magma_int_t r=0; r < U->num_rows; r++ ) {
        magma_int_t lasti = U->row[r];
        magma_int_t i=U->list[lasti];
        magma_int_t nexti=U->list[i];
        while( i!=0 ){
            if( MAGMA_Z_ABS( U->val[ i ] ) == MAGMA_Z_ZERO ){
                // the condition nexti!=0 ensures we never remove the diagonal
                    U->list[ i ] = -1;
                    if( U->col[ i ] == r ){
                        printf("error: try to rm diagonal.\n");   
                    }
                    
                    //omp_set_lock(&(counter));
                    //rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate
                    //printf("rm: (%d,%d) in location [%d]=%d\n", r, U->col[ i ], count_rm, i); fflush(stdout);
                    //count_rm++;
                    //omp_unset_lock(&(counter));
                    
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise

                    U->list[lasti] = nexti;
                    i = nexti;
                    nexti = U->list[nexti];
            }
            else{
                lasti = i;
                i = nexti;
                nexti = U->list[nexti];
            }
            
        }
    }
    /*
    
    #pragma omp parallel for
    for( magma_int_t rm=0; rm < *num_rm; rm++ ) {
        magma_int_t rm_row = LU_new->col[rm];
        magma_int_t rm_col = LU_new->rowidx[rm];
        magma_int_t lasti = U->row[rm_row];
        magma_int_t i=U->list[lasti];
        magma_int_t nexti=U->list[i];
        magma_int_t success = 0;
        while( success == 0 ){
            if( U->col[i] == rm_col ){
                omp_set_lock(&(counter));
                U->val[i] = MAGMA_Z_ZERO;
                rm_loc[ count_rm ] = i; 
                count_rm++;
                U->list[lasti] = nexti;
                i = nexti;
                nexti = U->list[nexti];
                omp_unset_lock(&(counter));
                success = 1;
            //    printf("done.\n"); fflush(stdout);
                    
            }
            //else if( U->col[i] > rm_col || i == 0 ){
            //    //printf("error:does not exist: %d (%d,%d)  (%d,%d) -> (%d,%d)\n", i, rm_row, rm_col, U->rowidx[lasti],U->col[lasti],U->rowidx[nexti],U->col[nexti]);fflush(stdout);
            //    omp_set_lock(&(counter));
            //    U->val[i] = MAGMA_Z_ZERO;
            //    rm_loc[ count_rm ] = i; 
            //    count_rm++;
            //    U->list[lasti] = nexti;
            //    i = nexti;
            //    nexti = U->list[nexti];
            //    omp_unset_lock(&(counter));
            //    success = 1;
            //}
            else{
                lasti = i;
                i = nexti;
                nexti = U->list[nexti];
            }
        }
    }
    */
    
      //  printf("second part done.\n"); fflush(stdout);
    //omp_destroy_lock(&(counter));
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
    L           magma_z_matrix*
                Current L approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    U           magma_z_matrix*
                Current U approximation where the identified smallest components
                are deleted.
                
    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
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
magma_zparilut_rm_thrs_LU(
    magmaDoubleComplex *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    omp_lock_t *rowlock,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    
    omp_lock_t counter;
    omp_init_lock(&(counter));
    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm
    magma_int_t offset = 0;
    
    //#pragma omp parallel for
    for( magma_int_t r=0; r < L->num_rows; r++ ) {
        magma_int_t i = L->row[r];
        magma_int_t lasti=i;
        magma_int_t nexti=L->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( L->val[ i ] ) <  MAGMA_Z_ABS(*thrs) ){
                // the condition nexti!=0 esures we never remove the diagonal
                    L->val[ i ] = MAGMA_Z_ZERO;
                    L->list[ i ] = -1;
                    if( L->col[ i ] == r ){
                        printf("error: try to rm diagonal.\n");   
                    }
                    
                    omp_set_lock(&(counter));
                    rm_loc[ count_rm ] = i; 
                    // keep it as potential fill-in candidate]
                    magma_int_t rm_col = L->col[ i ];
                    magma_int_t rm_row = r;
                    LU_new->col[ count_rm+offset ] = rm_col;
                    LU_new->rowidx[ count_rm+offset ] = r;
                    LU_new->val[ count_rm+offset ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
                    count_rm++;
                    omp_unset_lock(&(counter));
                    
                    magma_int_t i_U = i;
                    
                    // either the headpointer or the linked list has to be changed
                    // headpointer if the deleted element was the first element in the row
                    // linked list to skip this element otherwise
                    if( L->row[r] == i ){
                            L->row[r] = nexti;
                            lasti=i;
                            i = nexti;
                            nexti = L->list[nexti];
                    }
                    else{
                        L->list[lasti] = nexti;
                        i = nexti;
                        nexti = L->list[nexti];
                    }
                    
                    // now also update U
                    magma_int_t nexti_U = U->list[ i_U ];
                    if( U->row[ rm_col ] == i_U ){
                            U->row[ rm_col ] = nexti_U;
                    }
                    else{
                        while( nexti_U != 0 ){
                            if( U->col[ nexti_U ] == rm_row ){
                                U->list[ i_U ] = U->list[ nexti_U ];
                            }
                            i_U = nexti_U;
                            nexti_U = U->list[ i_U ];
                        }
                    }
                    
            }
            else{
                lasti = i;
                i = nexti;
                nexti = L->list[nexti];
            }
            
        }
    }
    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
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
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;;
    const magma_int_t ione = 1;
    
    CHECK( magma_zmalloc_cpu( &val, LU->nnz ));
    blasf77_zcopy(&LU->nnz, LU->val, &ione, val, &ione );
    if( order == 1 ){ // largest elements
      CHECK( magma_zorderstatistics(
        val, LU->nnz, num_rm, 1, &element, queue ) );
    } else if( order == 0) { // smallest elements
      CHECK( magma_zorderstatistics(
          val, LU->nnz, num_rm, 0, &element, queue ) );
    }
    
    *thrs = element;

cleanup:
    magma_free_cpu( val );
    return info;
}


/**
    Purpose
    -------
    This is a debugging routine, randomizing a linked list.

    Arguments
    ---------
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_randlist(
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex *val=NULL;
    magma_index_t *col=NULL, *rowidx=NULL, *row=NULL, *list=NULL, *randlist=NULL;
    
    magmaDoubleComplex *val_t=NULL;
    magma_index_t *col_t=NULL, *rowidx_t=NULL, *row_t=NULL, *list_t=NULL, *randlist_t=NULL;
    magma_index_t lasti = 0;
    magma_int_t nnz;
    
    CHECK( magma_index_malloc_cpu( &rowidx, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &list, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &row, LU->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &randlist, LU->true_nnz ));
    CHECK( magma_zmalloc_cpu( &val, LU->true_nnz ));
    
    //printf("nnz%d  true_nnz%d\t", LU->nnz, LU->true_nnz);
    // init list
    //#pragma omp parallel for
    for( magma_index_t i=0; i<LU->true_nnz; i++){
        randlist[i] = i;
        col[i] = -1;
    }
    //#pragma omp parallel for
    for( magma_index_t i=0; i<LU->true_nnz; i++){
        list[i] = -1;
        val[i] = MAGMA_Z_ZERO;
    }
    
    // randomize list
    for( magma_index_t i=0; i<LU->nnz*10000; i++){
        int loc1 = rand()%LU->row[LU->num_rows];
        int loc2 = rand()%LU->row[LU->num_rows];
        if(loc1 != loc2) {
            loc1=loc2;
            //printf ("changing locations: %d %d\n", loc1, loc2);
            int listt = randlist[loc1];
            randlist[loc1] = randlist[loc2];
            randlist[loc2] = listt;
        }
    }
    
    

   
    // now use this list to generate the new matrix structure
    nnz = 0;
    for( magma_index_t j=0; j<LU->num_rows; j++){    
        int i = LU->row[j];
        do{
            nnz++;
            int inew = randlist[i];
            //if( col[inew]!= -1 ){
            //    printf("error!\n");   
            //}
            val[ inew ] = LU->val[i];
            col[ inew ] = LU->col[i];
            rowidx[ inew ] = LU->rowidx[i];
            if( LU->row[j] == i ){
               row[j] = inew;
            } else {
              list[lasti] = inew;  
            }
            if( LU->list[i] == 0 ){
                list[inew] = 0;
            }
            lasti = inew;
            i=LU->list[i];
        }while( i!=0);
    }
    
    LU->nnz = nnz;
    row[LU->num_rows] = nnz;
    /*
    for( magma_index_t i=0; i<10; i++){
        
        printf("%d:%d (%d %d) %.2e \t", i, LU->list[i], LU->rowidx[i], LU->col[i], LU->val[i]);
        int newi = randlist[i];
        printf("%d:%d (%d %d) %.2e \n", newi, list[newi], rowidx[newi], col[newi], val[newi]);
        
    }
    
    for( magma_index_t i=0; i<LU->num_rows; i++){
        
        printf("%d:%d (%d %d) %.2e \t", i, LU->row[i],  LU->rowidx[LU->row[i]], LU->col[LU->row[i]], LU->val[LU->row[i]]);
        int newi = i;
        printf("%d:%d (%d %d) %.2e \n", i, row[newi],  rowidx[row[newi]], col[row[newi]], val[row[newi]]);

    }
    */
    //row[LU->num_rows] = LU->row[LU->num_rows];
   
    // flip
    val_t      = LU->val    ;
    col_t      = LU->col    ;
    row_t      = LU->row    ;
    rowidx_t   = LU->rowidx ;
    list_t     = LU->list   ;
    
    
    LU->val    = val        ;
    LU->col    = col        ;
    LU->row    = row        ;
    LU->rowidx = rowidx     ;
    LU->list   = list       ;
    

    val        = val_t      ;
    col        = col_t      ;
    row        = row_t      ;
    rowidx     = rowidx_t   ;
    list       = list_t     ;

cleanup:
    magma_free_cpu( rowidx );
    magma_free_cpu( row );
    magma_free_cpu( col );
    magma_free_cpu( list );
    magma_free_cpu( randlist );
    magma_free_cpu( val );
    return info;
}





/**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_set_approx_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;;
    const magma_int_t incy = 1;
    const magma_int_t incx = (int) (LU->nnz)/(1024);
    magma_int_t loc_nnz; 
    double ratio; 
    magma_int_t loc_num_rm; 
    magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    num_threads = 1;
    
    //printf("largest element considered: %d\n", LU->nnz);
    if( LU->nnz < 1024 ){
        loc_nnz = LU->nnz;
        ratio = ((double)num_rm)/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        CHECK( magma_zmalloc_cpu( &elements, num_threads ));
        blasf77_zcopy(&loc_nnz, LU->val, &incy, val, &incy );
        { 
            #pragma omp parallel
            {
                magma_int_t id = omp_get_thread_num();
                if(id<num_threads){
                    magma_zorderstatistics(
                        val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                }
            }
            element = MAGMA_Z_ZERO;
            for( int z=0; z<num_threads;z++){
                element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
            }
            element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
        } 
    } else {
        loc_nnz = (int) LU->nnz/incx;
        ratio = ((double)num_rm)/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
        { 
            if(num_threads > 1 ){
                CHECK( magma_zmalloc_cpu( &elements, num_threads ));
                #pragma omp parallel
                {
                    magma_int_t id = omp_get_thread_num();
                    if(id<num_threads){
                        magma_zorderstatistics(
                            val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                    }
                }
                element = MAGMA_Z_ZERO;
                for( int z=0; z<num_threads;z++){
                    element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
                }
                element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
            } else {
                magma_zorderstatistics(
                    val, loc_nnz, loc_num_rm, order, &element, queue );
            }
        } 
    }
    
    *thrs = element;

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
    return info;
}


/**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_LU_approx_thrs(
    magma_int_t num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;;
    const magma_int_t incy = 1;
    const magma_int_t incxL = (int) (L->nnz)/(1024);
    const magma_int_t incxU = (int) (U->nnz)/(1024);
    magma_int_t loc_nnz;
    magma_int_t loc_nnzL;
    magma_int_t loc_nnzU;
    double ratio; 
    magma_int_t loc_num_rm; 
    magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();   
    }
    num_threads = 1;
    
    {
        loc_nnz = (int) L->nnz/incxL + (int) U->nnz/incxU;
        loc_nnzL = (int) L->nnz/incxL;
        loc_nnzU = (int) U->nnz/incxU;
        ratio = ((double)num_rm)/((double)(L->nnz+U->nnz));
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        blasf77_zcopy(&loc_nnzL, L->val, &incxL, val, &incy );
        blasf77_zcopy(&loc_nnzU, U->val, &incxU, val+loc_nnzL, &incy );
        { 
            if(num_threads > 1 ){
                CHECK( magma_zmalloc_cpu( &elements, num_threads ));
                #pragma omp parallel
                {
                    magma_int_t id = omp_get_thread_num();
                    if(id<num_threads){
                        magma_zorderstatistics(
                            val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                    }
                }
                element = MAGMA_Z_ZERO;
                for( int z=0; z<num_threads;z++){
                    element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
                }
                element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
            } else {
                magma_zorderstatistics(
                    val, loc_nnz, loc_num_rm, order, &element, queue );
            }
        } 
    }
    
    *thrs = element;

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
    return info;
}



/**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.
                
    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.
                
    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.
                
    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/
/*
extern "C" magma_int_t
magma_zparilut_set_multi_thrs(
    magma_int_t num_thrs,
    magma_int_t *num_elements,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;;
    const magma_int_t incy = 1;
    const magma_int_t incx = (int) LU->nnz/1000;
    magma_int_t loc_nnz; 
    magma_int_t last_nnz; 
    double ratio; 
    magma_int_t loc_num_rm; 
    //printf("largest element considered: %d\n", LU->nnz);
    if( LU->nnz < 1000 ){
        CHECK( magma_zmalloc_cpu( &val, LU->nnz ));
        blasf77_zcopy(&loc_nnz, LU->val, &incy, val, &incy );
        CHECK( magma_zorderstatistics(
            val, LU->nnz, num_elements[0], order, &element[0], queue ) );
        for( int i=1; i<num_thrs; i++){ 
            CHECK( magma_zorderstatistics(
                val, num_elements[i-1], num_elements[i], order, &element[i], queue ) );
        }
    } else {
        loc_nnz = (int) LU->nnz/incx;
        ratio = ((double)num_elements[0])/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
        CHECK( magma_zorderstatistics(
            val, loc_nnz, loc_num_rm, order, &element[0], queue ) );
        for( int i=1; i<num_thrs; i++){ 
            last_nnz = loc_num_rm;
            loc_nnz = (int) LU->nnz/incx;
            ratio = ((double)num_elements[0])/((double)LU->nnz);
            loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
            CHECK( magma_zorderstatistics(
                val, last_nnz, loc_num_rm, order, &element[i], queue ) );
        }
    }
    
    *thrs = element;

cleanup:
    magma_free_cpu( val );
    return info;
}
*/



/**
    Purpose
    -------
    This function does an iterative ILU sweep.

    Arguments
    ---------
    
    @param[in]
    A           magma_int_t*
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
magma_zparict_sweep(
    magma_z_matrix *A,
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n");fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU->nnz; e++){
        if( LU->list[e]!=-1){
            magma_int_t i,j,icol,jcol,jold;
            
            magma_index_t row = LU->rowidx[ e ];
            magma_index_t col = LU->col[ e ];
            // as we look at the lower triangular, col<=row
            //printf("(%d,%d) ", row, col);fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    A->list[i] = 1;
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
        //printf("\n");fflush(stdout);
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
magma_zparict_residuals(
    magma_z_matrix A,
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<LU_new->nnz; e++){
        magma_int_t i,j,icol,jcol,jold;
        
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
            jold = j;
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
        LU_new->val[ e ] =  ( A_e - sum ) / LU.val[jold];
        
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
magma_zparict_candidates(
    magma_z_matrix LU,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    //LU_new->nnz = 0;
    
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
    //LU_new->nnz = 0;
    // should become fan-in
    numadd[ 0 ] = LU_new->nnz;
    for( magma_int_t i = 0; i<LU.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }

    if( LU_new->nnz > LU.nnz*5 ){
        printf("error: more candidates than space allocated. Increase candidate allocation.\n");
        goto cleanup;
    }
    
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
                    LU_new->val[ numadd[row] + ladd ] = MAGMA_Z_ZERO; // MAGMA_Z_MAKE(1e-14,0.0);
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



/**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.

    Arguments
    ---------
    
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmilu0_candidates(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *LU_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    LU_new->nnz = 0;
    
    magma_int_t also_lower = 0;
    
    magma_index_t *numadd;
    CHECK( magma_index_malloc_cpu( &numadd, L.num_rows+1 ));
    
    
    
    // additional step: all original matrix entries are candidates
    //
    /*
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numadd[i] = 0;  
    }

    // parallel loop counting elements
    #pragma omp parallel for
    for( magma_index_t row=0; row<A.num_rows; row++){
        magma_index_t start = A.row[ row ];
        magma_index_t end = A.row[ row+1 ];
        for( magma_index_t j=start; j<end; j++ ){
            if( A.list[j] == 0 ){
                numadd[ row+1 ]++;       
            }
        }
    }
    
    // get the total candidate count
    LU_new->nnz = 0;
    numadd[ 0 ] = LU_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }
    
    printf("%% candidates first step:%d\n", LU_new->nnz);
        
    // now insert
    #pragma omp parallel for
    for( magma_index_t row=0; row<A.num_rows; row++){
        magma_int_t ladd = 0;
        magma_index_t start = A.row[ row ];
        magma_index_t end = A.row[ row+1 ];
        for( magma_index_t j=start; j<end; j++ ){
            if( A.list[j] == 0 ){
                LU_new->val[ numadd[row] + ladd ] =  MAGMA_Z_MAKE(1e-14,0.0);
                LU_new->rowidx[ numadd[row] + ladd ] = row;
                LU_new->col[ numadd[row] + ladd ] = A.col[j];
                LU_new->list[ numadd[row] + ladd ] = -1;
                LU_new->row[ numadd[row] + ladd ] = -1;   
                ladd++;
            }
        }
    }
*/
    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numadd[i] = 0;  
    }
     
    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero

    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            magma_index_t start2 = U.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                el2 = U.list[ el2 ];
            }
            if( also_lower == 1 ){
            // try starting here
            col1 = L.col[ el1 ];
            start2 = L.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            }
            // end try here
            
            el1 = L.list[ el1 ];
        }
    } //loop over all rows
    
    
    // give it a show with ging the other way...
    /*
    // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<U.num_rows; row++){
        magma_index_t start1 = U.row[ row ];
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( U.list[el1] != 0 ) {
            magma_index_t col1 = U.col[ el1 ];
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//U.row[ cand_row ];//el1;
                magma_index_t checkcol = U.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = U.list[ checkel ];
                    checkcol = U.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            if( also_lower == 1 ){
            // try starting here
            col1 = U.col[ el1 ];
            start2 = U.row[ col1 ];
            el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//U.row[ cand_row ];//el1;
                magma_index_t checkcol = U.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        break;
                    }
                    checkel = U.list[ checkel ];
                    checkcol = U.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numadd[ row+1 ]++;
                }
                el2 = U.list[ el2 ];
            }
            }
            // end try here
            
            el1 = U.list[ el1 ];
        }
    } //loop over all rows
    */
    //#######################################
        
    // get the total candidate count
    //LU_new->nnz = 0;
    numadd[ 0 ] = LU_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        LU_new->nnz=LU_new->nnz + numadd[ i+1 ];
        numadd[ i+1 ] = LU_new->nnz;
    }
    // printf("cand count:%d\n", LU_new->nnz);
    if( LU_new->nnz > L.nnz*20 ){
        printf("error: more candidates than space allocated: %d>%d. Increase candidate allocation.\n", LU_new->nnz, L.nnz*20 );
        goto cleanup;
    }
        // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t ladd = 0;
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            magma_index_t start2 = U.row[ col1 ];
            magma_index_t el2 = start2;
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                         // printf("exist ---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                      // printf("---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = cand_row;
                    LU_new->col[ numadd[row] + ladd ] = cand_col;
                    LU_new->list[ numadd[row] + ladd ] = -1;
                    LU_new->row[ numadd[row] + ladd ] = -1;
                    ladd++;
                }
                el2 = U.list[ el2 ];
            }
            if( also_lower == 1 ){
            // try starting here
            col1 = L.col[ el1 ];
            start2 = L.row[ col1 ];
            el2 = start2;
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//L.row[ cand_row ];//el1;
                magma_index_t checkcol = L.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                         // printf("exist ---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                        exist = 1;
                        break;
                    }
                    checkel = L.list[ checkel ];
                    checkcol = L.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                      // printf("---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = cand_row;
                    LU_new->col[ numadd[row] + ladd ] = cand_col;
                    LU_new->list[ numadd[row] + ladd ] = -1;
                    LU_new->row[ numadd[row] + ladd ] = -1;
                    ladd++;
                }
                el2 = L.list[ el2 ];
            }
            }
            // end try here
            
            el1 = L.list[ el1 ];
        }
    } //loop over all rows
    
    
    // give it a shot with also the other way
    /*
            // parallel loop
    #pragma omp parallel for
    for( magma_index_t row=0; row<U.num_rows; row++){
        magma_index_t start1 = U.row[ row ];
        magma_int_t ladd = 0;
        
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        while( U.list[el1] != 0 ) {
            magma_index_t col1 = U.col[ el1 ];
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            while( L.list[ el2 ] != 0 ) {
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//U.row[ cand_row ];//el1;
                magma_index_t checkcol = U.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                         // printf("exist ---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                        exist = 1;
                        break;
                    }
                    checkel = U.list[ checkel ];
                    checkcol = U.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                      // printf("---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = cand_row;
                    LU_new->col[ numadd[row] + ladd ] = cand_col;
                    LU_new->list[ numadd[row] + ladd ] = -1;
                    LU_new->row[ numadd[row] + ladd ] = -1;
                    ladd++;
                }
                el2 = L.list[ el2 ];
            }
            if( also_lower == 1 ){
            // try starting here
            col1 = U.col[ el1 ];
            start2 = U.row[ col1 ];
            el2 = start2;
            while( U.list[ el2 ] != 0 ) {
                magma_index_t col2 = U.col[ el2 ];
                magma_index_t cand_row = max( row, col2);
                magma_index_t cand_col = min(row, col2);
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkel = el1;//U.row[ cand_row ];//el1;
                magma_index_t checkcol = U.col[ checkel ];
                while( checkcol <= cand_col && checkel!=0 ) {
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                         // printf("exist ---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                        exist = 1;
                        break;
                    }
                    checkel = U.list[ checkel ];
                    checkcol = U.col[ checkel ];
                }
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                      // printf("---------------->>>  candidate at (%d, %d)\n", cand_row, cand_col);
                    //add in the next location for this row
                    LU_new->val[ numadd[row] + ladd ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    LU_new->rowidx[ numadd[row] + ladd ] = cand_row;
                    LU_new->col[ numadd[row] + ladd ] = cand_col;
                    LU_new->list[ numadd[row] + ladd ] = -1;
                    LU_new->row[ numadd[row] + ladd ] = -1;
                    ladd++;
                }
                el2 = U.list[ el2 ];
            }
            }
            // end try here
            
            el1 = U.list[ el1 ];
        }
    } //loop over all rows
    */
    //#######################################

cleanup:
    magma_free_cpu( numadd );
    return info;
}





/**
    Purpose
    -------
    This routine converts a matrix back to CSR.
    Idea is to have a backup if the next iteration fails.

    Arguments
    ---------
                
    @param[in]
    LU          magma_z_matrix
                Current ILU approximation.
                
    @param[in,out]
    LU          magma_z_matrix*
                Matrix to fill.
                
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_copy(
    magma_z_matrix LU,
    magma_z_matrix *LUCSR,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // rowpointer
    #pragma omp parallel for
    for(magma_int_t i=0; i<LU.num_rows+1; i++){
        LUCSR->row[i] = LU.row[ i ];    
    }
    // col index
    #pragma omp parallel for
    for(magma_int_t i=0; i<LU.nnz; i++){
        LUCSR->col[i] = LU.col[ i ];    
    }
    // val
    #pragma omp parallel for
    for(magma_int_t i=0; i<LU.nnz; i++){
        LUCSR->val[i] = LU.val[ i ];    
    }

    return info;
}


/**
    Purpose
    -------
    This routine sets the list etries of A to zero.

    Arguments
    ---------
                
    @param[in]
    A           magma_z_matrix*
                Current ILU approximation.
                
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_zero(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // rowpointer
    #pragma omp parallel for
    for(magma_int_t i=0; i<A->num_rows; i++){
        for( magma_int_t j=A->row[i]; j<A->row[i+1]; j++ ){
            A->list[j] = 0;    
        }   
    }

    return info;
}


magma_int_t
magma_zdiagcheck_cpu(
    magma_z_matrix A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // rowpointer
    #pragma omp parallel for
    for(magma_int_t i=0; i<A.num_rows; i++){
        magmaDoubleComplex diag = MAGMA_Z_ZERO;
        for( magma_int_t j=A.row[i]; j<A.row[i+1]; j++ ){
            if( A.col[j] == i ){
                diag = A.val[j];
            }   
        }
        if( diag == MAGMA_Z_ZERO ){
            printf("%%error: zero diagonal element in row %d.\n", i );    
            info = i;
            for( magma_int_t l=A.row[i]; l<A.row[i+1]; l++ ){
                printf("(%d,%d)%.2e -> ", i, A.col[l], A.val[l]);
            }   
            printf("\n\n");
        }
    }
    return info;
}





#endif  // _OPENMP
