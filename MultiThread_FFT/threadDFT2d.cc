// Threaded two-dimensional Discrete FFT transform
// Seth Stewart
// ECE8893 Project 2


#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>
#include "Complex.h"
#include "InputImage.h"
#include "GoodBarrier.h"

#define DEBUG 1


// Need global variables indicating how
// many threads there are, and a Complex* that points to the
// 2d image being transformed.

using namespace std;

int num_threads = 16;
int barray[16];
int height,width, N;
Complex* h;
Complex* weights;
Complex* iweights;
int startCount;
pthread_mutex_t h_mutex, weight_mutex, iweight_mutex, exit_mutex, exe_mutex, cout_mutex, b_mutex;
//GoodBarrier barrier;
pthread_barrier_t barrier;
pthread_cond_t exit_c;


// Function to reverse bits in an unsigned integer
// This assumes there is a global variable N that is the
// number of points in the 1D transform.
unsigned ReverseBits(unsigned v)
{ //  Provided to students
  unsigned n = N; // Size of array (which is even 2 power k value)
  unsigned r = 0; // Return value
   
  for (--n; n > 0; n >>= 1)
    {
      r <<= 1;        // Shift return value
      r |= (v & 0x1); // Merge in next bit
      v >>= 1;        // Shift reversal value
    }
  return r;
}

void BitReverseElements(Complex* H){
	for (unsigned i=0;i < width;i++)
	{
		unsigned j = ReverseBits(i);
		if(j<i)
		{
			Complex temp = H[i];
			H[i] = H[j];
			H[j] = temp;
		}
	}
}

int ArraySum()
{
  int j = 0;
  for (int i = 0;i < num_threads; i++ )
  {
      j = j+ barray[i];
  }
  return j;
}
void ArrayReset()
{
   for (int i = 0;i<num_threads;i++)
   {
      barray[i]=0;
   }
}

void CalculateWeights(Complex* weights, int N)
{
	//Precalculate weighting values
	//Implements (from 0 to N - 1 of e^-j2*pi*k/N) weights
	//cout << "weight n = " << N << endl;
	for(int i = 0; i < N; i++)
	{
		//double real, imag;
		
		Complex c(cos(2 * M_PI * ((double)i) / double(width)), -sin(2 * M_PI * ((double)i) / double(width)));
		//real = cos(2 * M_PI * ((double)i) / double(width));
		//imag = -sin(2 * M_PI * ((double)i) / double(width));
	 	
		//cout << "arg = "<< 2*M_PI*((double)i/width)<<endl;
		//cout << "imag = " << imag << endl;
	
		weights[i] = c;
		//weights[i].real = real;
		if (fabs(weights[i].real) < 1e-10) weights[i].real = 0;
		//weights[i].imag = imag;
		if (fabs(weights[i].imag) < 1e-10) weights[i].imag = 0;
		
	//	cout << "Weight [" << i << "] =" << weights[i].imag<<endl;	
	}
	cout << "Weights [ ";
	for (int i=0; i<N; i++){
		if (i != N-1){
			cout<<weights[i]<< ", ";
		}
		else{
			cout << weights[i];
			cout<<" ]" <<endl;
		}
	}
}

void CalculateIWeights(Complex* weights, int N)
{
	//Precalculate weighting values
	//Implements (from 0 to N - 1 of e^-j2*pi*k/N) weights
	//cout << "weight n = " << N << endl;
	for(int i = 0; i < N; i++)
	{
		//double real, imag;
		
		Complex c(cos(2 * M_PI * ((double)i) / double(width)), +sin(2 * M_PI * ((double)i) / double(width)));
		//real = cos(2 * M_PI * ((double)i) / double(width));
		//imag = +sin(2 * M_PI * ((double)i) / double(width));
	 	
		//cout << "arg = "<< 2*M_PI*((double)i/width)<<endl;
		//cout << "imag = " << imag << endl;
	
		weights[i] = c;
		//weights[i].real = real;
		if (fabs(weights[i].real) < 1e-10) weights[i].real = 0;
		//weights[i].imag = imag;
		if (fabs(weights[i].imag) < 1e-10) weights[i].imag = 0;
		
	//	cout << "Weight [" << i << "] =" << weights[i].imag<<endl;	
	}
	cout << "IWeights [ ";
	for (int i=0; i<N; i++){
		if (i != N-1){
			cout<<weights[i]<< ", ";
		}
		else{
			cout << weights[i];
			cout<<" ]" <<endl;
		}
	}
}

// GRAD Students implement the following 2 functions.
// Undergrads can use the built-in barriers in pthreads.

// Call MyBarrier_Init once in main
void MyBarrier_Set(int N)// you will likely need some parameters)
{
   barray[N] = 1;
}
 
// Each thread calls MyBarrier after completing the row-wise DFT
void MyBarrier() // Again likely need parameters
{
  while(ArraySum() != num_threads)
  { }
}
        
void Transform1D(Complex* h, int n)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)

	//Complex* bitRevRow = new Complex[N];
	//cout << "Bit Reversing" << endl;
	pthread_mutex_lock(&h_mutex);
		BitReverseElements(h);
	pthread_mutex_unlock(&h_mutex);
	//cout<< "n = "<< n;
	//cout << "Bit Reversing Complete" << endl;
	//
//for(int count = start; count <(rows*width + start); count+=width){
	for(int points=2; points <= n; points *= 2){	//powers of 2 chunk size
		pthread_mutex_lock(&cout_mutex);
			//cout << "\t" << "Starting " << points <<" point transform"<<endl;
		pthread_mutex_unlock(&cout_mutex);
		//cout << points << " point transform"<< endl;
		for (int chunk=0; chunk < n/points; chunk++){	//chunk/transform number
			pthread_mutex_lock(&cout_mutex);
				//cout << "\n Chunk " << chunk << ", ";
			pthread_mutex_unlock(&cout_mutex);
	
			for (int element=0; element < points/2; element++){ //element in the transform
				Complex temp = h[element+(chunk*points)];
				/*
				pthread_mutex_lock(&cout_mutex);
					cout<< "element = "<<element<<endl;
					cout << "temp: h[" << element+chunk*points <<"] = " << h[element+chunk*points] << endl;
					pthread_mutex_lock(&weight_mutex);
						cout<< 	"w[" << element*n/points <<  "] = "<< weights[element*n/points]<<endl;
						cout << "h[" <<element+(points*chunk)<<"] = " << "temp + w[" << element*n/points<< "]*h["<< (points*chunk)+element+points/2<<"]"<< "= "<<h[element+(points*chunk)]<< endl;
						cout << "h[" <<element+(points*chunk)+(points/2) << "] = temp - w["<< (element*n)/points<<"]*" << "h[" <<element+(chunk*points)+points/2<<"]"<< "= "<< h[element+(points*chunk)+points/2]<< "\n"<< endl;
					pthread_mutex_unlock(&weight_mutex);
				pthread_mutex_unlock(&cout_mutex);
				*/
				pthread_mutex_lock(&weight_mutex);		
					h[element+(points*chunk)] = temp + weights[element*n/points]*h[(points*chunk)+element+points/2];
					h[element+(points*chunk)+points/2] = temp - weights[element*n/points]*h[element+(chunk*points)+points/2];
				pthread_mutex_unlock(&weight_mutex);
				
				if (fabs(h[element+(points*chunk)].real) < 1e-10) h[element+(points*chunk)].real= 0;
				if (fabs(h[element+(points*chunk)].imag) < 1e-10) h[element+(points*chunk)].imag = 0;
				if (fabs(h[element+(points*chunk)+points/2].real) < 1e-10) h[element+(points*chunk)+points/2].real = 0;
				if (fabs(h[element+(points*chunk)+points/2].imag) < 1e-10) h[element+(points*chunk)+points/2].imag = 0;
			}
		}
	}
	
	//Try to fix the elements (Not sure why they are wrong, according to my print statements, they should be right.
	for (int i = 0; i <n; i++)
	{
		if (i!=0 && i!= n/2)
		{
			h[i].real = h[i].real*(n/2);
			h[i].imag = h[i].imag*(n/2);
		}
		else if (i==n/2)
		{
			h[i].real = h[i].real/(n/2);
			h[i].imag = h[i].imag/(n/2);
		}
		else{}
	}
		
//}
}

void Inverse1D(Complex* h, int n)
{
  // Implement the efficient Danielson-Lanczos DFT here.
  // "h" is an input/output parameter
  // "N" is the size of the array (assume even power of 2)

	//Complex* bitRevRow = new Complex[N];
	//cout << "Bit Reversing" << endl;
	pthread_mutex_lock(&h_mutex);
		BitReverseElements(h);
	pthread_mutex_unlock(&h_mutex);
	//cout<< "n = "<< n;
	//cout << "Bit Reversing Complete" << endl;
	//
//for(int count = start; count <(rows*width + start); count+=width){
	for(int points=2; points <= n; points *= 2){	//powers of 2 chunk size
		pthread_mutex_lock(&cout_mutex);
			//cout << "\t" << "Starting " << points <<" point transform"<<endl;
		pthread_mutex_unlock(&cout_mutex);
		//cout << points << " point transform"<< endl;
		for (int chunk=0; chunk < n/points; chunk++){	//chunk/transform number
			pthread_mutex_lock(&cout_mutex);
				//cout << "\n Chunk " << chunk << ", ";
			pthread_mutex_unlock(&cout_mutex);
	
			for (int element=0; element < points/2; element++){ //element in the transform
				Complex temp = h[element+(chunk*points)];
				/*
				pthread_mutex_lock(&cout_mutex);
					cout<< "element = "<<element<<endl;
					cout << "temp: h[" << element+chunk*points <<"] = " << h[element+chunk*points] << endl;
					pthread_mutex_lock(&iweight_mutex);
						cout<< 	"iw[" << element*n/points <<  "] = "<< iweights[element*n/points]<<endl;
						cout << "h[" <<element+(points*chunk)<<"] = " << "temp + iw[" << element*n/points<< "]*h["<< (points*chunk)+element+points/2<<"]"<< "= "<<h[element+(points*chunk)]<< endl;
						cout << "h[" <<element+(points*chunk)+(points/2) << "] = temp - iw["<< (element*n)/points<<"]*" << "h[" <<element+(chunk*points)+points/2<<"]"<< "= "<< h[element+(points*chunk)+points/2]<< "\n"<< endl;
					pthread_mutex_unlock(&iweight_mutex);
				pthread_mutex_unlock(&cout_mutex);
				*/
				pthread_mutex_lock(&iweight_mutex);		
					h[element+(points*chunk)] = temp + iweights[element*n/points]*h[(points*chunk)+element+points/2];
					h[element+(points*chunk)+points/2] = temp - iweights[element*n/points]*h[element+(chunk*points)+points/2];
				pthread_mutex_unlock(&iweight_mutex);
				
				if (fabs(h[element+(points*chunk)].real) < 1e-10) h[element+(points*chunk)].real= 0;
				if (fabs(h[element+(points*chunk)].imag) < 1e-10) h[element+(points*chunk)].imag = 0;
				if (fabs(h[element+(points*chunk)+points/2].real) < 1e-10) h[element+(points*chunk)+points/2].real = 0;
				if (fabs(h[element+(points*chunk)+points/2].imag) < 1e-10) h[element+(points*chunk)+points/2].imag = 0;
			}
		}
	}
	//Try to fix the elements
	for (int i = 0; i <n; i++)
	{
		if (i!=0 && i!= n/2)
		{
			h[i].real = h[i].real*(n/2);
			h[i].imag = h[i].imag*(n/2);
		}
		else if (i==n/2)
		{
			h[i].real = h[i].real/(n/2);
			h[i].imag = h[i].imag/(n/2);
		}
		else{}
	}
		
//}
}

void Transpose (Complex* in, Complex* out, int iw){
	for (int i=0; i<iw ; ++i){
		for (int j=0; j<iw ; ++j){
			out[i*iw+j] = in[i+j*iw];
		}
	}
}

void* Transform2DThread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  
	long thread = (long) v;
	
	//pthread_mutex_lock(&cout_mutex);
		//cout << "thread = " << thread << endl;
	//pthread_mutex_unlock(&cout_mutex);
	
	pthread_mutex_lock(&h_mutex);
		if (num_threads <= 1) num_threads = 1;
		int rowsPerThread = height / num_threads;
		//pthread_mutex_lock(&cout_mutex);
				//cout << "num_threads = " << num_threads << endl;
				//cout << "rowsPerThread = " << rowsPerThread << endl;
		//pthread_mutex_unlock(&cout_mutex);
	pthread_mutex_unlock(&h_mutex);

	int startRow = (thread) * rowsPerThread;
	int stopRow = (thread+1) * rowsPerThread;
	
	pthread_mutex_lock(&h_mutex);
		Complex* row = new Complex[width];
	pthread_mutex_unlock(&h_mutex);

	//cout << "rowsPerThread = " << rowsPerThread << endl;
	//pthread_mutex_lock(&cout_mutex);
		//cout << "startRow = " << startRow << endl;
		//cout << "stopRow = " << stopRow << endl;
	//pthread_mutex_unlock(&cout_mutex);
	
	for (int i=startRow; i<stopRow; i++)
	{
		pthread_mutex_lock(&h_mutex);
			for(int j=0; j<width;j++)
			{
				//cout << "j = " << j <<endl;
				//should not need mutex. Each thread grabbing different rows
				row[j]=h[i*height+j];	//grab a row for 1D
			}
		pthread_mutex_unlock(&h_mutex);
		
		pthread_mutex_lock(&cout_mutex);
			cout << "transforming Row " << i << "\n"<<endl;
		pthread_mutex_unlock(&cout_mutex);

		Transform1D(row,width);		//do the 1D
		
		
			for (int j=0;  j<width; j++)
			{
				pthread_mutex_lock(&h_mutex);
				h[i*height+j] = row[j]; 	//put it back in the image
				pthread_mutex_unlock(&h_mutex);
			}
		
		//grab the next row (i++)
	}

  // wait for all to complete
	pthread_mutex_lock(&exe_mutex);
	
		//pthread_mutex_lock(&cout_mutex);
			//cout<<"\t"<<"Decrementing startCount"<<endl;
		//pthread_mutex_unlock(&cout_mutex);
		
		startCount--;
		if (startCount == 0)
		{ // Last to exit, notify main
			
			//debug statement
			pthread_mutex_lock(&cout_mutex);
				cout<<"\t"<<"startCount == 0"<<endl;
			pthread_mutex_unlock(&cout_mutex);
			
			//reset startcount number
			pthread_mutex_lock(&h_mutex);
				startCount = num_threads;
			pthread_mutex_unlock(&h_mutex);
			
	pthread_mutex_unlock(&exe_mutex);
			
			//lock before signal
			pthread_mutex_lock(&exit_mutex);
				//debug
				pthread_mutex_lock(&cout_mutex);
					cout<<"\t"<<"signaling main"<<endl;
				pthread_mutex_unlock(&cout_mutex);
				//signal
				pthread_cond_signal(&exit_c);
			//unlock after signal
			pthread_mutex_unlock(&exit_mutex);
		}
		else
		{
			pthread_mutex_unlock(&exe_mutex);
		}

 /* 
  pthread_mutex_lock(&b_mutex);
  MyBarrier_Set(thread);
  pthread_mutex_unlock(&b_mutex);
*/
  return 0;
}

void* Inverse2DThread(void* v)
{ // This is the thread startign point.  "v" is the thread number
  // Calculate 1d DFT for assigned rows
  
	long thread = (long) v;
	
	//pthread_mutex_lock(&cout_mutex);
		//cout << "thread = " << thread << endl;
	//pthread_mutex_unlock(&cout_mutex);
	
	pthread_mutex_lock(&h_mutex);
		if (num_threads <= 1) num_threads = 1;
		int rowsPerThread = height / num_threads;
		//pthread_mutex_lock(&cout_mutex);
				//cout << "num_threads = " << num_threads << endl;
				//cout << "rowsPerThread = " << rowsPerThread << endl;
		//pthread_mutex_unlock(&cout_mutex);
	pthread_mutex_unlock(&h_mutex);

	int startRow = (thread) * rowsPerThread;
	int stopRow = (thread+1) * rowsPerThread;
	
	pthread_mutex_lock(&h_mutex);
		Complex* row = new Complex[width];
	pthread_mutex_unlock(&h_mutex);

	//cout << "rowsPerThread = " << rowsPerThread << endl;
	//pthread_mutex_lock(&cout_mutex);
		//cout << "startRow = " << startRow << endl;
		//cout << "stopRow = " << stopRow << endl;
	//pthread_mutex_unlock(&cout_mutex);
	
	for (int i=startRow; i<stopRow; i++)
	{
		pthread_mutex_lock(&h_mutex);
			for(int j=0; j<width;j++)
			{
				//cout << "j = " << j <<endl;
				//should not need mutex. Each thread grabbing different rows
				row[j]=h[i*height+j];	//grab a row for 1D
			}
		pthread_mutex_unlock(&h_mutex);
		
		pthread_mutex_lock(&cout_mutex);
			cout << "Inverting Row " << i << "\n"<<endl;
		pthread_mutex_unlock(&cout_mutex);

		Inverse1D(row,width);		//do the 1D inverse
		
		
			for (int j=0;  j<width; j++)
			{
				pthread_mutex_lock(&h_mutex);
				h[i*height+j] = row[j]; 	//put it back in the image
				pthread_mutex_unlock(&h_mutex);
			}
		
		//grab the next row (i++)
	}

  // wait for all to complete
  pthread_mutex_lock(&b_mutex);
  MyBarrier_Set(thread);
  pthread_mutex_unlock(&b_mutex);
	
  return 0;
}

void Transform2D(const char* inputFN) 
{ // Do the 2D transform here.
	InputImage image(inputFN);  // Create the helper object for reading the image
	height = image.GetHeight();
	width = image.GetWidth();
	N = height * width;
	cout << "N = " << N << endl;
	
	// Create the global pointer to the image array data
	h = image.GetImageData();
	
	//Make and calculate the weight array
	weights = new Complex[width/2];
	CalculateWeights(weights, width/2);
	
	//Transform2DThread((void*)1 , inputFN);
	
  	// Create 16 threads
	for(int i = 0; i < num_threads; i++)
	{
		pthread_t thread;
		pthread_create(&thread, 0, Transform2DThread, (void*)i);
	}	
 	
 	// Wait for all threads complete
 	pthread_cond_wait(&exit_c, &exit_mutex);
	//MyBarrier();
	//ArrayReset();
	
	cout << "1D complete, Writing to 1d.txt"<<endl;
	image.SaveImageData("Tower-DFT1D.txt", h, width, height);
	
	//Dont need a mutex here, because this isnt threaded
	Complex* Transposed = new Complex[N];
	Transpose(h, Transposed, width);

	//Put the transpose back into
	for (int i = 0; i<height; i++)
	{
		for (int j =0; j<width; j++)
		{
			h[i*height+j] = Transposed[i*height+j];
		}
	}
	
	cout << "Transpose complete, Continuing to 2D"<<endl;

	//Repeat the process for the columns
	for (int i =0; i<num_threads; i++)
	{ 
		pthread_t thread;
		pthread_create(&thread, 0, Transform2DThread, (void*)i);
	}
	
	pthread_cond_wait(&exit_c, &exit_mutex);
    //MyBarrier();
    //ArrayReset();
    cout<<"\n2D Complete, Starting Inverse\n"<<endl;
	
  	
  	// Write the transformed data
  	image.SaveImageData("Tower-DFT2D.txt", h, width, height); 
}

void Inverse2D(const char* inputFN) 
{ // Do the 2D transform here.
	InputImage image(inputFN);  // Create the helper object for reading the image
	height = image.GetHeight();
	width = image.GetWidth();
	N = height * width;
	//cout << "N = " << N << endl;
	
	// Create the global pointer to the image array data
	h = image.GetImageData();
	
	//Make and calculate the weight array
	iweights = new Complex[width/2];
	CalculateIWeights(iweights, width/2);
	//pthread_mutex_lock(&exit_mutex);
	//Transform2DThread((void*)1 , inputFN);
	
  	// Create 16 threads
	for(int i = 0; i < num_threads; i++)
	{
		pthread_t thread;
		pthread_create(&thread, 0, Inverse2DThread, (void*)i);
	}	
 	// Wait for all threads complete
	MyBarrier();
    ArrayReset();
	
	cout << "Inverse 1D complete\n"<<endl;
	image.SaveImageData("Inverse1d.txt", h, width, height);
	
	Complex* Transposed = new Complex[N];
	Transpose(h, Transposed, width);

	//Put the transpose back into h
	for (int i = 0; i<height; i++)
	{
		for (int j =0; j<width; j++)
		{
			h[i*height+j] = Transposed[i*height+j];
		}
	}

	//Repeat the process for the columns
	for (int i =0; i<num_threads; i++)
	{ 
		pthread_t thread;
		pthread_create(&thread, 0, Inverse2DThread, (void*)i);
	}
	
	MyBarrier();
    ArrayReset();
    
	//pthread_cond_wait(&exit_c, &exit_mutex);
	
	for(int i=0; i<N; i++)
	{
		h[i].real = h[i].real/N;
		h[i].imag = h[i].imag/N;
	}
	
  	// Write the transformed data
  	image.SaveImageData("TowerInverse.txt", h, width, height); 
}

int main(int argc, char** argv)
{
  pthread_mutex_init(&h_mutex, 0);
  pthread_mutex_init(&weight_mutex, 0);
  pthread_mutex_init(&iweight_mutex, 0);
  pthread_mutex_init(&exe_mutex, 0);
  pthread_mutex_init(&exit_mutex, 0);
  pthread_mutex_init(&cout_mutex, 0);
  pthread_mutex_init(&b_mutex, 0);
  pthread_cond_init(&exit_c, 0);
  string fn("Tower.txt"); // default file name
  if (argc > 1){
	fn = string(argv[1]);  // if name specified on cmd line
	num_threads = atoi (argv[2]);
  }
  startCount=num_threads;

  ArrayReset();
  Transform2D(fn.c_str()); // Perform the transform.
  Inverse2D("Tower-DFT2D.txt");
}  
