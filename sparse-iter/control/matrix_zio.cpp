/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       June 2013

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
//extern "C"{
#include "../include/mmio.h"
//}


using namespace std;


magma_int_t read_z_csr_from_binary(magma_int_t* n_row, magma_int_t* n_col, magma_int_t* nnz, magmaDoubleComplex **val, magma_int_t **col, magma_int_t **row, const char * filename){


  std::fstream binary_test(filename);


  if(binary_test){
    printf("#Start reading...");
    fflush(stdout);
  }
  else{
    printf("#Unable to open file ", filename);
    fflush(stdout);
    exit(1);
  }
  binary_test.close();
  
  
  std::fstream binary_rfile(filename,std::ios::binary|std::ios::in);
  
  
  //read number of rows
  binary_rfile.read(reinterpret_cast<char *>(n_row),sizeof(int));
  
  //read number of columns
  binary_rfile.read(reinterpret_cast<char *>(n_col),sizeof(int));
  
  //read number of nonzeros
  binary_rfile.read(reinterpret_cast<char *>(nnz),sizeof(int));
  
  
  *val = new magmaDoubleComplex[*nnz];
  *col = new int[*nnz];
  *row = new int[*n_row+1];
  
  
  //read row pointer
  for(magma_int_t i=0;i<=*n_row;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*row)[i]),sizeof(int));
  }
  
  //read col
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*col)[i]),sizeof(int));
  }
  
  //read val
  for(magma_int_t i=0;i<*nnz;i++){
    binary_rfile.read(reinterpret_cast<char *>(&(*val)[i]),sizeof(double));
  }

  binary_rfile.close();
  
  printf("#Finished reading.");
  fflush(stdout);
}

magma_int_t read_z_csr_from_mtx(magma_int_t* n_row, magma_int_t* n_col, magma_int_t* nnz, magmaDoubleComplex **val, magma_int_t **col, magma_int_t **row, const char *filename){
  
  FILE *fid;
  MM_typecode matcode;
    
  fid = fopen(filename, "r");
  
  if (fid == NULL) {
    printf("#Unable to open file %s\n", filename);
    exit(1);
  }
  
  if (mm_read_banner(fid, &matcode) != 0) {
    printf("#Could not process lMatrix Market banner.\n");
    exit(1);
  }
  
  if (!mm_is_valid(matcode)) {
    printf("#Invalid lMatrix Market file.\n");
    exit(1);
  }
  
  if (!((mm_is_real(matcode) || mm_is_integer(matcode) || mm_is_pattern(matcode)) && mm_is_coordinate(matcode) && mm_is_sparse(matcode))) {
    printf("#Sorry, this application does not support ");
    printf("#Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    printf("#Only sparse real-valued or pattern coordinate matrices are supported\n");
    exit(1);
  }
  
  magma_int_t num_rows, num_cols, num_nonzeros;
  if (mm_read_mtx_crd_size(fid,&num_rows,&num_cols,&num_nonzeros) !=0)
    exit(1);
  
  (*n_row) = (magma_int_t) num_rows;
  (*n_col) = (magma_int_t) num_cols;
  (*nnz)   = (magma_int_t) num_nonzeros;

  magma_int_t *coo_col, *coo_row;
  magmaDoubleComplex *coo_val;
  
  coo_col = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t));
  assert(coo_col != NULL);

  coo_row = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t)); 
  assert( coo_row != NULL);

  coo_val = (magmaDoubleComplex *) malloc(*nnz*sizeof(magmaDoubleComplex));
  assert( coo_val != NULL);


  printf("#Reading sparse matrix from file (%s):",filename);
  fflush(stdout);


  if (mm_is_real(matcode) || mm_is_integer(matcode)){
    for(magma_int_t i = 0; i < *nnz; ++i){
      magma_int_t ROW ,COL;
      double VAL;  // always read in a double and convert later if necessary
      
      fscanf(fid, " %d %d %lf \n", &ROW, &COL, &VAL);   
      
      coo_row[i] = (magma_int_t) ROW - 1; 
      coo_col[i] = (magma_int_t) COL - 1;
      coo_val[i] = MAGMA_Z_MAKE( VAL, 0.);
    }
  } else {
    printf("Unrecognized data type\n");
    exit(1);
  }
  
  fclose(fid);
  printf(" done\n");
  
  

  if(mm_is_symmetric(matcode)) { //duplicate off diagonal entries
  printf("#symmetric!!!");
    magma_int_t off_diagonals = 0;
    for(magma_int_t i = 0; i < *nnz; ++i){
      if(coo_row[i] != coo_col[i])
        ++off_diagonals;
    }
    
    magma_int_t true_nonzeros = 2*off_diagonals + (*nnz - off_diagonals);
    
    
    cout<<"#nnz="<<*nnz<<endl;

    
    
    magma_int_t* new_row = (magma_int_t *) malloc(true_nonzeros*sizeof(magma_int_t)) ; 
    magma_int_t* new_col = (magma_int_t *) malloc(true_nonzeros*sizeof(magma_int_t)) ; 
    magmaDoubleComplex* new_val = (magmaDoubleComplex *) malloc(true_nonzeros*sizeof(magmaDoubleComplex)) ; 
    
    magma_int_t ptr = 0;
    for(magma_int_t i = 0; i < *nnz; ++i) {
        if(coo_row[i] != coo_col[i]) {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
        new_col[ptr] = coo_row[i];  
        new_row[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;  
      } else 
      {
        new_row[ptr] = coo_row[i];  
        new_col[ptr] = coo_col[i];  
        new_val[ptr] = coo_val[i];
        ptr++;
      }
    }      
    
    free (coo_row);
    free (coo_col);
    free (coo_val);

    coo_row = new_row;  
    coo_col = new_col; 
    coo_val = new_val;   
    
    *nnz = true_nonzeros;
    

  } //end symmetric case
  
  magmaDoubleComplex tv;
  magma_int_t ti;
  
  
  //If matrix is not in standard format, sorting is necessary
  /*
  
    std::cout << "Sorting the cols...." << std::endl;
  // bubble sort (by cols)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if (coo_col[j] > coo_col[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }

  std::cout << "Sorting the rows...." << std::endl;
  // bubble sort (by rows)
  for (int i=0; i<*nnz-1; ++i)
    for (int j=0; j<*nnz-i-1; ++j)
      if ( coo_row[j] > coo_row[j+1] ){

        ti = coo_col[j];
        coo_col[j] = coo_col[j+1];
        coo_col[j+1] = ti;

        ti = coo_row[j];
        coo_row[j] = coo_row[j+1];
        coo_row[j+1] = ti;

        tv = coo_val[j];
        coo_val[j] = coo_val[j+1];
        coo_val[j+1] = tv;

      }
  std::cout << "Sorting: done" << std::endl;
  
  */
  
  
  (*val) = (magmaDoubleComplex *) malloc(*nnz*sizeof(magmaDoubleComplex)) ;
  assert((*val) != NULL);
  
  (*col) = (magma_int_t *) malloc(*nnz*sizeof(magma_int_t));
  assert((*col) != NULL);
  
  (*row) = (magma_int_t *) malloc((*n_row+1)*sizeof(magma_int_t)) ;
  assert((*row) != NULL);
  

  
  
  // original code from  Nathan Bell and Michael Garland
  // the output CSR structure is NOT sorted!

  for (magma_int_t i = 0; i < num_rows; i++)
    (*row)[i] = 0;
  
  for (magma_int_t i = 0; i < *nnz; i++)
    (*row)[coo_row[i]]++;
  
  
  //cumsum the nnz per row to get Bp[]
  for(magma_int_t i = 0, cumsum = 0; i < num_rows; i++){     
    magma_int_t temp = (*row)[i];
    (*row)[i] = cumsum;
    cumsum += temp;
  }
  (*row)[num_rows] = *nnz;
  
  //write Aj,Ax into Bj,Bx
  for(magma_int_t i = 0; i < *nnz; i++){
    magma_int_t row_  = coo_row[i];
    magma_int_t dest = (*row)[row_];
    
    (*col)[dest] = coo_col[i];
    
    (*val)[dest] = coo_val[i];
    
    (*row)[row_]++;
  }
  
  for(int i = 0, last = 0; i <= num_rows; i++){
    int temp = (*row)[i];
    (*row)[i]  = last;
    last   = temp;
  }
  
  (*row)[*n_row]=*nnz;
     

  for (magma_int_t k=0; k<*n_row; ++k)
    for (magma_int_t i=(*row)[k]; i<(*row)[k+1]-1; ++i) 
      for (magma_int_t j=(*row)[k]; j<(*row)[k+1]-1; ++j) 

      if ( (*col)[j] > (*col)[j+1] ){

        ti = (*col)[j];
        (*col)[j] = (*col)[j+1];
        (*col)[j+1] = ti;

        tv = (*val)[j];
        (*val)[j] = (*val)[j+1];
        (*val)[j+1] = tv;

      }


}



magma_int_t write_z_csr_mtx(magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, magmaDoubleComplex **val, magma_int_t **col, magma_int_t **row, const char *filename)
{
   std::ofstream file(filename);
   file << "%%MatrixMarket matrix coordinate real general" << std::endl;
   file << n_row <<" "<< n_col <<" "<< nnz << std::endl;
   
  magma_int_t i=0,j=0;

  for(i=0; i<n_col; i++)
  {
    magma_int_t rowtemp1=(*row)[i];
    magma_int_t rowtemp2=(*row)[i+1];
    for(j=0; j<rowtemp2-rowtemp1; j++)  
      {
  //b[i*n+col[rowtemp1+j]]=val[rowtemp1+j];
  file << (rowtemp1+1) <<" "<< ((*col)[rowtemp1+j]+1) <<" "<< MAGMA_Z_REAL((*val)[rowtemp1+j]) << std::endl;
      }
  }
}

magma_int_t cout_z_csr_mtx(magma_int_t n_row, magma_int_t n_col, magma_int_t nnz, magmaDoubleComplex **val, magma_int_t **col, magma_int_t **row)
{
   cout << "%%MatrixMarket matrix coordinate real general" << endl;
   cout << n_row <<" "<< n_col <<" "<< nnz <<endl;
   
  magma_int_t i=0,j=0;

  for(i=0; i<n_col; i++)
  {
    magma_int_t rowtemp1=(*row)[i];
    magma_int_t rowtemp2=(*row)[i+1];
    for(j=0; j<rowtemp2-rowtemp1; j++)  
      {
  //b[i*n+col[rowtemp1+j]]=val[rowtemp1+j];
  cout<< rowtemp1+1 <<" "<< (*col)[rowtemp1+j]+1 <<" "<< MAGMA_Z_REAL((*val)[rowtemp1+j]) <<endl;
      }
  }
}
