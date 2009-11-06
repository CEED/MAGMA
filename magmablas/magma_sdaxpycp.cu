extern "C" __global__ void sdaxpycp_special(float *R, double *X, int M, int ldr,int lda, double *B,double *W ) {
	const int ibx = blockIdx.x * 64;
	const int idt = threadIdx.x;
	X+= ibx+idt;
        R+= ibx+idt;
	B+= ibx+idt;
	W+= ibx+idt;
        X[0]= X[0]+(double) R[0];
	W[0] = B[0];
}
extern "C" __global__ void sdaxpycp_generic(float *R, double *X, int M, int ldr,int lda, double *B,double *W ) {
	const int ibx = blockIdx.x * 64;
	const int idt = threadIdx.x;
	if( ( ibx + idt ) < M ) {
		X+= ibx+idt;
	       	R+= ibx+idt;
		B+= ibx+idt;
		W+= ibx+idt;
	}
	else{
		X+=(M-1);
		R+=(M-1);
		B+=(M-1);
		W+=(M-1);
	}
        X[0]= X[0]+(double) R[0];
	W[0] = B[0];
}
extern "C" void magmablas_sdaxpycp(float *R, double *X, int M, int ldr,int lda, double *B, double *W) 
{    
        dim3 threads( 64, 1 );
        dim3 grid(M/64+(M%64!=0),1);
	if( M %64 == 0 ) {
        	sdaxpycp_special <<< grid, threads >>> ( R, X, M, ldr,lda, B, W) ;
	}
	else{
        	sdaxpycp_generic <<< grid, threads >>> ( R, X, M, ldr,lda, B, W) ;
	}
}           
