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
#include "Complex.cu"

#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            (512)
// Maximum iterations for the MBSet calculations
#define MAX_IT                (2000)

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;

// Block and thread counts for CUDA
dim3 blocks(WINDOW_DIM/8, WINDOW_DIM/8);
dim3 threads(8, 8);




_global__ void calculateInSet(Complex * minC, Complex * maxC, int * iterations, int * result)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int width = blockDim.x * gridDim.x;
	int height = blockDim.y * gridDim.y;
	int offset = x + y * width;

	Complex complexOffset(x * (maxC->r - minC->r)/width, y * (maxC->i - minC->i)/height);
	Complex c(*minC + complexOffset);

	int maxIteration = *iterations;
	int i = 0;
	Complex z(c);
	while(i != maxIteration)
	{
		if(z.magnitude2() > 4.0f)
		{
			break;
		}
		z = z*z + c;
		++i;
	}

	result[offset] = i;


}

void  computeMandelBrotSet()
{

	cudaMalloc((void **) &dev_minC, sizeof(Complex));

	cudaMalloc((void **) &dev_maxC, sizeof(Complex));
	cudaMalloc((void **) &dev_iterations, sizeof(int));
	cudaMalloc((void **) &dev_iterationArray, sizeof(int) * WINDOW_DIM * WINDOW_DIM);

	cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_iterations, &iterations, sizeof(int), cudaMemcpyHostToDevice);

	calculateInSet<<<blocks, threads>>>(dev_minC, dev_maxC, dev_iterations, dev_iterationArray);

	cudaMemcpy(host_iterationArray, dev_iterationArray, sizeof(int) * WINDOW_DIM * WINDOW_DIM, cudaMemcpyDeviceToHost);
	return;
}

int width=512, height=512; // window size
int windowID;

GLfloat minX = -2.0f, maxX = 1.0f, minY = -1.2f, maxY = 1.8f; // complex plane boundaries
GLfloat stepX = (maxX - minX)/(GLfloat)width;
GLfloat stepY = (maxY - minY)/(GLfloat)height;

GLfloat black[] = {0.0f, 0.0f, 0.0f}; // black color
GLfloat white[] = {1.0f, 1.0f, 1.0f}; // white color
const int paletteSize = 128;
GLfloat palette[paletteSize][3];

bool fullScreen=false;

double maxIteration = 2000;

double startMouseClickX = 0.0;
double startMouseClickY = 0.0;
double endMouseClickX = 0.0;
double endMouseClickY = 0.0;
double currentMouseX = 0.0;
double currentMouseY = 0.0;
bool isBox = false;

void drag (int x, int y){
	//	cout << "============================="<<endl;
	//	cout << x << '\t' <<y<< endl;
	currentMouseX = x;
	currentMouseY = y;

}

void mouse(int button, int state, int x, int y){
	//	cout << "============================="<<endl;

	if (state==GLUT_DOWN)	{
		cout << "DOWN" <<endl;
		//		cout << "x: " << x << "\n";
		//		cout << "y: " << y << "\n";
		startMouseClickX = x;
		startMouseClickY = y;


	}
	if (state==GLUT_UP)	{
		cout << "UP" <<endl;

		//		cout << "x: " << x << "\n";
		//		cout << "y: " << y << "\n";
		endMouseClickX = x;
		endMouseClickY = y;
		isBox = true;
		cout << "Redisplaying" <<endl;
		glutPostRedisplay();
		isBox = false;
	}
}


GLfloat* calculateColor(GLfloat u, GLfloat v){
	GLfloat re = u;
	GLfloat im = v;
	GLfloat tempRe=0.0;
	Complex c = Complex((float)re,(float)im);
	Complex Zn0 = c;
	Complex Zn1(0,0);
	bool isWhite = false;
	short isWhiteIter = -100;

	for (int i = 0; i < maxIteration; ++i) {
		Zn1 = Zn0*Zn0 + c;
		if (Zn1.magnitude2() > 2.0*2.0) {
			isWhite = true;
			isWhiteIter = i;
			break;
			cout << "breaking!!";
		}
		Zn0 = Zn1;
	}

	if(isWhite && isWhiteIter >= 0)	{
		return palette[isWhiteIter%128];
	}
	else return black;
}


GLfloat* mandelImage[512][512];

void repaint() {// function called to repaint the window
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen buffer
	glBegin(GL_POINTS); // start drawing in single pixel mode

	short cx = 0;
	short cy = 0;

	for(GLfloat y = maxY; y >= minY; y -= stepY){
		for(GLfloat x = minX; x <= maxX; x += stepX){
			glColor3fv(mandelImage[cx][cy]); // set color

			glVertex2f(cx,cy);
			cx++;
		}
		cy++;
		cx = 0;
	}
	glEnd(); // end drawing
	isBox = true;
	if (isBox) {
		//		float side = ((endMouseClickX - startMouseClickX) < (endMouseClickY - startMouseClickY)) ? endMouseClickX - startMouseClickX : endMouseClickY - startMouseClickY;
		//		endMouseClickX = startMouseClickX + side;
		//		endMouseClickY = startMouseClickY + side;
		//		float topLeftXTransformed = (startMouseClickX - 256.0)/256.0;
		//		float topLeftYTransformed = (256.0 - startMouseClickY)/256.0;
		//
		//		float bottomRightXTransformed = (endMouseClickX - 256.0)/256.0;
		//		float bottomRightYTransformed = (256.0 - endMouseClickY)/256.0;
		//
		//		cout<<"Drawing red box: ("<<topLeftXTransformed<<", "<<topLeftYTransformed<<") : (" << bottomRightXTransformed <<", "<< bottomRightYTransformed <<")\n";
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINE_LOOP);
		//		glVertex2f(topLeftXTransformed, topLeftYTransformed);
		//		glVertex2f(bottomRightXTransformed,topLeftYTransformed);
		//		glVertex2f(bottomRightXTransformed,bottomRightYTransformed);
		//		glVertex2f(topLeftXTransformed,bottomRightYTransformed);
		glVertex2f(0.5,0.5);
		glVertex2f(0.25,0.25);
		glEnd();
	}

	glutSwapBuffers(); // swap the buffers - [ 2 ]
}

void createPalette(){
	int eight = 4;
	int four = 2;
	for(int i=0; i < 32; i++){
		palette[i][0] = (eight*i)/(GLfloat)255;
		palette[i][1] = (128-four*i)/(GLfloat)255;
		palette[i][2] = (255-eight*i)/(GLfloat)255;
	}
	for(int i=0; i < 32; i++){
		palette[32+i][0] = (GLfloat)1;
		palette[32+i][1] = (eight*i)/(GLfloat)255;
		palette[32+i][2] = (GLfloat)0;
	}
	for(int i=0; i < 32; i++){
		palette[64+i][0] = (128-four*i)/(GLfloat)255;
		palette[64+i][1] = (GLfloat)1;
		palette[64+i][2] = (eight*i)/(GLfloat)255;
	}
	for(int i=0; i < 32; i++){
		palette[96+i][0] = (GLfloat)0;
		palette[96+i][1] = (255-eight*i)/(GLfloat)255;
		palette[96+i][2] = (eight*i)/(GLfloat)255;
	}
}




int main(int argc, char** argv)
{
	// Initialize OPENGL here
	// Set up necessary host and device buffers
	// set up the opengl callbacks for display, mouse and keyboard

	// Calculate the interation counts
	// Grad students, pick the colors for the 0 .. 1999 iteration count pixels

	glutInit(&argc, argv);
	createPalette();
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	GLsizei windowX = (glutGet(GLUT_SCREEN_WIDTH)-width)/2;
	GLsizei windowY = (glutGet(GLUT_SCREEN_HEIGHT)-height)/2;
	glutInitWindowPosition(windowX, windowY);
	glutInitWindowSize(width, height);
	windowID = glutCreateWindow("MANDELBROTH");

	glShadeModel(GL_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	//	glViewport (0, 0, (GLsizei) width, (GLsizei) height);
	//	glMatrixMode (GL_PROJECTION);
	//	glLoadIdentity();
	//	glOrtho(minX, maxX, minY, maxY, ((GLfloat)-1), (GLfloat)1);

	// set the event handling methods

	short cx = 0;
	short cy = 0;
	for(GLfloat y = maxY; y >= minY; y -= stepY){
		for(GLfloat x = minX; x <= maxX; x += stepX){
			GLfloat* temp  = calculateColor(x,y);
			//			cout << temp;
			//			cout << cx <<"\t"<< cy <<endl;
			mandelImage[cx][cy] = temp;
			cx++;
		}
		cy++;
		cx = 0;
	}

	glutDisplayFunc(repaint);
	//	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyFunction);
	glutMouseFunc(mouse);
	glutMotionFunc(drag);


	glutMainLoop(); // THis will callback the display, keyboard and mouse
	return 0;

}