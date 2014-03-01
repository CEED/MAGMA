// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>
#include <time.h>

#include "/home/hanzt/software/fastsparse/spmvELLRTdp.h"

// includes, project
#include "flops.h"
#include "magma.h"
#include "../include/magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"
#include "mkl_spblas.h"



#define MKL_ADDR(a) (a)


void cudaInit(int dev)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);                                                         
    /*CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceCount(&deviceCount));                
    if (deviceCount == 0) {                                                  
        fprintf(stderr, "cutil error: no devices supporting CUDA.\n");       
        exit(EXIT_FAILURE);                                                  
    }                                                                        
    */
    //if (dev < 0) dev = cutGetMaxGflopsDeviceId();

    if (dev > deviceCount-1) dev = deviceCount - 1;                          
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);                                               
    /*CUDA_SAFE_CALL_NO_SYNC(cudaGetDeviceProperties(&deviceProp, dev));       
    if (deviceProp.major < 1) {                                              
        fprintf(stderr, "cutil error: device does not support CUDA.\n");     
        exit(EXIT_FAILURE);                                                  
    }               */                                                         
    fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);  
    cudaSetDevice(dev);     
    //CUDA_SAFE_CALL(cudaSetDevice(dev));                                 
}


int main(int argc,char *argv[]){


   char *filename[] =
    {
    "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/pwtk.mtx",
    "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/cant.mtx",
    "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/cage10.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/jj_ML_Geer.mtx",
    // "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/circuit5M.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/jj_Cube_Coup_dt0.mtx",
   //  "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/Serena_b.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/dielFilterV2real.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/Hook_1498.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/bone010.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/boneS10.mtx",
   //  "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/FullChip.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/audikw_1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/bmwcra_1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/crankseg_2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/F1.mtx",   
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/inline_1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/ldoor.mtx",
//     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/af_shell10.mtx",
//     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/kim2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/Fault_639.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/StocF-1465.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/kkt_power.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/bmwcra_1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/thermal2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/m_t1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/pre2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/bmw3_2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/xenon2.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/stomach.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/shipsec1.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/airfoil_2d.mtx",
     "/home/hanzt/magma_dev/magma/trunk/sparse-iter/testing/test_matrices/Trefethen_20000.mtx"};

for(int matrix=0; matrix<40; matrix++){

double *A;           /** Sparse matrix **/
double *v;               /** Vector to multiply **/
double *u;               /** Result **/
unsigned int *index;    /** Indices matrix **/
unsigned int *col;      /** Row-length **/ 

double       *BA;
unsigned int *BIndex;
unsigned int *BCol;

int blocksize,device,symmetric,threads,ones,texture;

char *file;       /* Sparse matrix file */
FILE *pf;         /* Pointer to the file */
char line[1024];   /* Complete line of the file */
float value;      /* Value of the element i,j  */

  
           file=filename[matrix];
      blocksize=atoi(argv[2]);   
        threads=atoi(argv[3]);  
         device=atoi(argv[4]);
        texture=atoi(argv[5]);

  cudaInit(device);

  int nrows,ncols,nzeros,alignment,nmaxblocks;
  size_t size_a,size_v,size_u,size_i,size_c;
  int i,j,max,min;

  /** Get the sparse matrix from the input file **/

    if((pf = fopen(file,"r"))==NULL){
      printf("Can't open the file: %s",file);
      exit(1);
  }

        char banner[64];
        char mtx[64];   
        char crd[64];
        char data_type[64];
        char storage_scheme[64];
 
      fgets(line,sizeof(line),pf);                 /* First line: matrix name */
    
        sscanf(line,"%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme);    

        ones=0;symmetric=0;

        if (strcmp(data_type,"pattern")==0) ones=1;    
     
        if (strcmp(storage_scheme,"symmetric")==0) symmetric=1;

        fgets(line,sizeof(line),pf); 
        //    printf("ignore: %s", line);
          while(line[0] == '%'){
            fgets(line,sizeof(line),pf); 
        //    printf("ignore: %s", line);
        }
   
        sscanf(line,"%d %d %d",&nrows,&ncols,&nzeros);  /* Second line: rows, cols, non zeros=entries */
     
        printf("\nSym: %d. Pattern: %d. Nrows: %d. Ncols: %d. Entries: %d",symmetric,ones,nrows,ncols,nzeros);
        printf("\n");
        printf("\n");


        /* First step. Calculate the max column */

        size_c=nrows*sizeof(unsigned int);
        col=(unsigned int *)malloc(size_c);

        for(i=0;i<nrows;i++)
          col[i]=0;

        while (!feof(pf)){
            fgets(line,sizeof(line),pf); 
            //printf("print1 %s", line);
              while(line[0] == '%')
                fgets(line,sizeof(line),pf); 
            //printf("print2 %s", line);
        if (ones==0){
            sscanf(line,"%d %d %f",&i,&j,&value);
        }//fscanf(pf,"%d %d %f",&i,&j,&value);
        
          else if(ones==1)sscanf(line,"%d %d",&i,&j);
          else sscanf(line,"%f %d %d",&value,&i,&j);
       
          if(ones<2){
         i--;  /*Begin from 1*/ 
       j--;
          }

      col[i]++;
          if((symmetric==1)&&(i!=j))col[j]++;
        }
       
    fclose(pf);



        max=0;
        min=ncols;
        for(i=0;i<nrows;i++){
          if(col[i]>max)max=col[i];
          if(col[i]<min)min=col[i];
        }


        size_a=max*nrows*sizeof(double);
        size_i=max*nrows*sizeof(unsigned int);
               
        A=(double *)malloc(size_a);
        index=(unsigned int *)malloc(size_i);
        
        /* Initalization of A,index with zeros */

        for(i=0;i<nrows*max;i++){
               A[i]=0.0; 
               index[i]=0; 
               if (i<nrows)col[i]=0;
        }


        /* Second step: the values of the matrix */

        if((pf = fopen(file,"r"))==NULL){
         printf("Can't open the file: %s",file);
         exit(1);
        }
  
      fgets(line,200,pf);              
        fscanf(pf,"%d %d %d",&nrows,&ncols,&nzeros);                

        nzeros=0;

        value=1.0;
        while (!feof(pf)){
            fgets(line,sizeof(line),pf); 
            //printf("print1 %s", line);
              while(line[0] == '%')
                fgets(line,sizeof(line),pf); 
               if(ones==0)sscanf(line,"%d %d %f",&i,&j,&value);
          else if(ones==1)sscanf(line,"%d %d",&i,&j);
          else sscanf(line,"%f %d %d",&value,&i,&j);
       
          if(ones<2){
         i--;  /*Begin from 1 */
       j--;
          }

      A[i*max+col[i]]=value;
          index[i*max+col[i]]=j;
          col[i]++;
          nzeros++;
          if((symmetric==1)&&(i!=j)){
             
                A[j*max+col[j]]=value;
                index[j*max+col[j]]=i;
                col[j]++;
                nzeros++;
          }


    }
    fclose(pf);




        size_v=ncols*sizeof(double);
        size_u=nrows*sizeof(double);
        
        v=(double *)malloc(size_v);
        u=(double *)malloc(size_u);



        srand48(time(NULL));        
        for(i=0;i<nrows;i++){
         if (i<ncols)v[i]=1.0;
         u[i]=0.0;
        }


        /* Matrix BA, BIndex, BCol */

        alignment=ceil((float)(nrows*threads)/128)*128;
        nmaxblocks=ceil((float)max/threads);

        size_t size_ba=nmaxblocks*alignment*sizeof(double);
        size_t size_bi=nmaxblocks*alignment*sizeof(unsigned int);

        BA=(double *)malloc(size_ba);       /* Matrix in ELLRT format */
        BIndex=(unsigned int *)malloc(size_bi); /* Index columns */
        BCol=(unsigned int *)malloc(size_c); /* Row length */

        for(i=0;i<nmaxblocks*alignment;i++){
          BA[i]=0.0;
          BIndex[i]=0;
          if (i<nrows)BCol[i]=ceil((float)col[i]/threads);
        }

        int k;
 
        for(i=0;i<nrows;i++){
          j=0;
          k=0;
          while(j<col[i]){
           for(int m=0;m<threads;m++){
             if(j<col[i]){
               BA[k*alignment+(i*threads)+m]=A[i*max+j];
               BIndex[k*alignment+(i*threads)+m]=index[i*max+j];
             }
             j++;
           }
           k++;
          }
        }


   /* Device's structures */

   double *d_BA;cudaMalloc((void**)&d_BA,size_ba);
   double *d_v;cudaMalloc((void**)&d_v,size_v);
   double *d_u;cudaMalloc((void**)&d_u,size_u);
   unsigned int *d_BIndex;cudaMalloc((void**)&d_BIndex,size_bi);
   unsigned int *d_BCol;cudaMalloc((void**)&d_BCol,size_c);


   /* Copy data to GPU */
   cudaMemcpy(d_BA,BA,size_ba,cudaMemcpyHostToDevice);   
   cudaMemcpy(d_BIndex,BIndex,size_bi,cudaMemcpyHostToDevice);
   cudaMemcpy(d_BCol,BCol,size_c,cudaMemcpyHostToDevice);

   cudaMemcpy(d_v,v,size_v,cudaMemcpyHostToDevice);   
   cudaMemcpy(d_u,u,size_u,cudaMemcpyHostToDevice);   


   /** Library **/
    double start, end;
    start = magma_wtime(); 

    for (i=0; i<1; i++ )
        spmv_ELLRTdp(d_BA,d_v,d_u,d_BIndex,d_BCol,nrows,alignment,blocksize,threads,texture);

    end = magma_wtime();
    printf( " > SpMVELLRT  : %.2e seconds (ELLRT).\n",(end-start)/10 );


  /* Copy data from GPU to host */ 

   cudaMemcpy(u,d_u,size_u,cudaMemcpyDeviceToHost);   


// MKL comparson//
        magma_d_sparse_matrix hA, hB, hC, dA, dB, hD, dD, hE, dE;
        magma_d_vector hx, hy, dx, dy, dx2, dy2;
        double one = 1.0;
        double zero = 0.0;

  
        // init matrix on CPU
        magma_d_csr_mtx( &hA, filename[matrix] );
        //printf( "\nmatrix read: %d-by-%d with %d nonzeros\n\n",hA.num_rows,hA.num_cols,hA.nnz );

        // init CPU vectors
        magma_d_vinit( &hx, Magma_CPU, hA.num_rows, 1.0 );
        magma_d_vinit( &hy, Magma_CPU, hA.num_rows, 0.0 );

        //printf( " num_rblocks=%d, block_size=%d\n",num_rblocks,hC.blocksize );

        // calling MKL with CSR
        int *pntre;
        pntre = (int*)malloc( (hA.num_rows+1)*sizeof(int) );
        pntre[0] = 0;
        for (i=0; i<hA.num_rows; i++ ) pntre[i] = hA.row[i+1];
        start = magma_wtime(); 
        for (i=0; i<1; i++ )
        mkl_dcsrmv( "N", &hA.num_rows, &hA.num_cols, 
                    MKL_ADDR(&one), "GFNC", MKL_ADDR(hA.val), hA.col, hA.row, pntre, 
                                            MKL_ADDR(hx.val), 
                    MKL_ADDR(&zero),        MKL_ADDR(hy.val) );
        end = magma_wtime();
        printf( "\n > MKL  : %.2e seconds (CSR).\n",(end-start)/10 );
        //printf( "  %.2e  &",(end-start)/10 );
        free(pntre);

        for(int i=0; i<hA.num_rows; i++)
            if(abs(hy.val[i]-u[i])>0.001 ){
                printf("error in index %d : %f vs %f difference. %e\n", i, hy.val[i],u[i], hy.val[i]-u[i]);
            }

    // MKL comparson//


   // Free memory in the GPU

   cudaFree(d_BA);
   cudaFree(d_v);
   cudaFree(d_u);
   cudaFree(d_BIndex);
   cudaFree(d_BCol);

   // Free memory in the host

   free(A);
   free(index);
   free(BA);
   free(BIndex);
   free(col);
   free(BCol);
   free(v);
   free(u);



     printf("\n***");
     printf("\n   Algorithm: ELLRT double precision. T = %d. BS = %d. Alignment = %d",threads,blocksize,alignment);
     printf("\n   Memory: %2.4f MB. Matrix name: %s. NonZeros: %d %dx%d",(size_ba+size_v+size_u+size_bi+size_c)/1048576.0,file,nzeros,nrows,ncols);
     printf("\n***");
     printf("\n");

}// end matrix

}






