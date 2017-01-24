/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <vector>
#include "Complex.cu"
#include <unistd.h>
#include <cuda_runtime_api.h>

#include <GL/freeglut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>


// Size of window in pixels, both width and height
#define WINDOW_DIM            512
//threads per block
#define THREADS_PB	      32
#define N		      (WINDOW_DIM*WINDOW_DIM)

using namespace std;

//==================GLOBALS========================

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
const int maxIt = 2000; // Msximum Iterations

float xmin = -2.0, xmax = 1.0;
float ymin = -1.2, ymax = 1.8;

Complex* c = new Complex[WINDOW_DIM * WINDOW_DIM];
Complex* dev_c;       

int w = WINDOW_DIM;
int h = WINDOW_DIM;

int iter_count [N];
int* dev_icount;

float dx, dy, dz;

bool drawing = false;


//===============CLASSES/STUCTS==================

class RGB
{
  public: RGB(): r(0), g(0), b(0){ }
  RGB( float r0, float g0, double b0): r(r0),g(g0),b(b0){}
  public: 
  	float r;
	float g;
	float b;
};

struct Point
{
  int x,y;
};

struct Frame
{
  public: 
  float minC_x;
  float minC_y;
  float maxC_x;
  float maxC_y;
  Frame(float a, float b, float c, float d) : minC_x(a), minC_y(b), maxC_x(c), maxC_y(d){}
};

//============CLASS VARIABLES===================

vector <Frame> frame_vec;
RGB* colors = 0; // Array of color values
Point start, end;

//============ MB FUNCTIONS ========================

__global__ void calcMB (Complex* dev_minC, Complex* dev_maxC, int* dev_icount, Complex* dev_c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int i = index / WINDOW_DIM;
  int j = index % WINDOW_DIM; 
  double dr = dev_maxC->r - dev_minC->r;
  double di = dev_maxC->i - dev_minC->i;
  double nr = (double) i / WINDOW_DIM;
  double ni = (double) j / WINDOW_DIM;
	
  dev_c[index].r = dev_minC->r + nr * dr;
  dev_c[index].i = dev_minC->i + ni * di;

  Complex Z (0,0);
  Z.r = dev_c[index].r;
  Z.i = dev_c[index].i;
  dev_icount[index] = 0;
  
  while(dev_icount[index] < maxIt)
  {
	if (Z.magnitude2() < 4.0f)
	{
		dev_icount[index]++;
		Z = (Z*Z) + dev_c[index];
	}
	else{ break; }
  }	
}

void cuda()
{
  //make space on the device
  cudaMalloc((void**)&dev_icount, N *sizeof(int));
  cudaMalloc((void**)&dev_c, N * sizeof(Complex));
  cudaMalloc((void**)&dev_minC, sizeof(Complex));
  cudaMalloc((void**)&dev_maxC, sizeof(Complex));
  //copy from host to device
  cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_icount, iter_count, N*sizeof(int), cudaMemcpyHostToDevice);  
  cudaMemcpy(dev_c, c, N*sizeof(Complex), cudaMemcpyHostToDevice);  
  //do the calculation
  calcMB<<< N / THREADS_PB, THREADS_PB >>>(dev_minC, dev_maxC, dev_icount, dev_c);
  //copy from device to host
  cudaMemcpy(iter_count, dev_icount, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(c, dev_c, N*sizeof(Complex), cudaMemcpyDeviceToHost);
}

//================ OGL FUNCTIONS ==================
void Square()
{
  glColor3f(1, 1, 1);
  glBegin(GL_LINE_LOOP);
  glVertex2f(start.x, end.y);
  glVertex2f(start.x, start.y);
  glVertex2f(end.x, start.y);
  glVertex2f(end.x, end.y);
  glEnd(); 
}

void display()
{
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, WINDOW_DIM, WINDOW_DIM, 0);
  glDisable(GL_DEPTH_TEST);
  glMatrixMode(GL_MODELVIEW);
  glClearColor(0.0, 0.0, 0.0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  
  glBegin(GL_POINTS);
  for(int i = 0; i < WINDOW_DIM; i++)
  {
    for(int j = 0; j < WINDOW_DIM; j++)
    {
      glColor3f(colors[iter_count[i*WINDOW_DIM + j]].r, colors[iter_count[i*WINDOW_DIM + j]].g, colors[iter_count[i*WINDOW_DIM + j]].b);
      glVertex2d(i, j);
    } 
  }
  glEnd();
  
  if(drawing)
    Square();
  glutSwapBuffers();
}

//================ USER INPUT ==================

void mouse(int button, int state, int x, int y)
{
  	if(button == GLUT_LEFT_BUTTON){
		if(state==GLUT_DOWN) 
		{
			start.x = x; end.x = x;
			start.y = y; end.y = y;
			drawing = true;		
		}
		
		if(state==GLUT_UP) {
			frame_vec.push_back(Frame(minC.r, minC.i, maxC.r, maxC.i));
			if(x > start.x && y > start.y)
			{
			  end.x = start.x + dz;
			  end.y = start.y + dz;
			}
			else if(x < start.x && y < start.y)
			{
			  end.x = start.x - dz;
			  end.y = start.y - dz;
			}
			else if(x > start.x && y < start.y)
			{
			  end.x = start.x + dz;
			  end.y = start.y - dz;
			}
			else if(x < start.x && y > start.y)
			{
			  end.x = start.x - dz;
			  end.y = start.y + dz;
			}
			
			for(int i = 0; i < WINDOW_DIM; i++)
			{
				for(int j = 0; j < WINDOW_DIM; j++)
				{
					if(i == start.x && j  == start.y)
					{
						minC.r = c[i*WINDOW_DIM + j].r;
						minC.i = c[i*WINDOW_DIM + j].i;
					}

					if(i == end.x && j  == end.y)
					{
						maxC.r = c[i*WINDOW_DIM + j].r;
						maxC.i = c[i*WINDOW_DIM + j].i;
					}
				}
			}
			cuda();
			drawing = false;
			glutPostRedisplay();
		}
	}
}

void motion(int x, int y)                                                              
{
  dx = abs(x - start.x);
  dy = abs(y - start.y);
  if(dx > dy) dz = dy;
  if(dx < dy) dz = dx;
  if(x > start.x && y > start.y)
  {

    end.x = start.x + dz;  
    end.y = start.y + dz;
  }

  if(x < start.x && y < start.y)
  {
    end.x = start.x - dz;  
    end.y = start.y - dz;
  }

  if(x < start.x && y > start.y)
  {
    end.x = start.x - dz;  
    end.y = start.y + dz;
  }
  
  if(x > start.x && y < start.y)
  {
    end.x = start.x + dz;  
    end.y = start.y - dz;
  }
 
  glutPostRedisplay();
}


void keyboard(unsigned char key, int x, int y)
{
  if (key == 'b')
  {
    //load previous frame
    if(frame_vec.size() > 0 )
    {
      Frame back = frame_vec.back();
      frame_vec.pop_back();
      cout<<"Vector size = "<<frame_vec.size()<<endl;                            
      minC.r = back.minC_x;
      minC.i = back.minC_y;
      maxC.r = back.maxC_x;
      maxC.i = back.maxC_y;
      cuda();                                                               
      glutPostRedisplay();                                                               
    }
  }
  if (key =='q')
  {
	freeMem();
	exit(0);
  }
}

void InitializeColors()
{
  time_t t = time(0);
  srand48(t);
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 5)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
          colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}

void freeMem()
{
    free(c);
    cudaFree(d_icount);
    cudaFree(d_minC);
	cudaFree(dev_c);
    cudaFree(d_maxC);
    cout << "Exited Cleanly!" << endl;
}

int main(int argc, char** argv)
{
  // Initialize OPENGL here
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels

  InitializeColors();
  glutInit(&argc, argv);
  //Window
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
  glutInitWindowPosition(100,100);
  glutCreateWindow("MBSet");
  
  cuda();

  //Callbacks
  glutDisplayFunc(display);
  glutIdleFunc(display);
  glutMouseFunc(mouse);
  glutKeyboardFunc(keyboard);
  glutMotionFunc(motion);

  glutMainLoop(); // This will callback the display, keyboard and mouse
  return 0;
  
}
