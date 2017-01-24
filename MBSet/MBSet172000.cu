//@ex172000 ex172000 Final
#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdarg.h>
#include <vector>
#include <ctime>
#include <math.h>
#include "Complex.cu"
#include <GL/freeglut.h>
#include <algorithm>
#define  N (512 * 512)
#define  THREADS_PER_BLOCK 32
#define  maxIt 2000

using namespace std;

bool  click = false;
bool  xflag = false;
bool  yflag = false;
int   WINDOW_DIM = 512;
int   *n,  *d_n;
int   *id, *d_id;
float *xm, *d_xm, *ym, *d_ym;
float *xa, *d_xa, *ya, *d_ya;
float xmin = -2.0, xmax = 1.0;
float ymin = -1.2, ymax = 1.8;
float minxs, maxxs, minys, maxys;
vector<float> xminv, yminv, xmaxv, ymaxv;

class RGB {
	public:
  	RGB() : r(0), g(0), b(0) {}
  	RGB(float r0, float g0, float b0) : r(r0), g(g0), b(b0) {}
  	float r;
  	float g;
  	float b;
};

struct po { float x; float y; };
struct nh { int nx[N]; };
struct nh no;
vector<nh> nhistory;
po st, fh, tp, br;
RGB* colors = 0; 

__global__ void pointcalc(int *n, float *xm, float *ym, float *xa, float *ya) {
  	int index = threadIdx.x + blockIdx.x * blockDim.x;
	float aa = 0.0, bb = 0.0; n[index] = 0;
	while ( (aa * aa + bb * bb) < 4 && n[index] < maxIt) {
	  	float tmp = aa * aa - bb * bb; 
	  	bb = 2.0 * aa * bb + *ym + ((float)(index / 512) / (float)512) * (*ya - *ym);
	  	aa = tmp           + *xm + ((float)(index % 512) / (float)512) * (*xa - *xm);
	  	n[index] = n[index] + 1;
	}
}

void InitializeColors() {
	time_t t = time(0);
	srand48(t);
	colors = new RGB[maxIt + 1];
	for (int i = 0; i < maxIt; ++i) {
    	if (i <  5)  
          	colors[i] = RGB(1, 1, 1);
        if (i >= 5) 
          	colors[i] = RGB(drand48(), drand48(), drand48());
    }
  	colors[maxIt] = RGB(); 
}

void display() {
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, 0, 1);
	glDisable(GL_DEPTH_TEST);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(.0, .0, .0, 0);
	glClear(GL_COLOR_BUFFER_BIT);   
    cudaMalloc((void **)&d_n, N * sizeof(int));
    cudaMalloc((void **)&d_xm,  sizeof(float));
    cudaMalloc((void **)&d_ym,  sizeof(float));
    cudaMalloc((void **)&d_xa,  sizeof(float));
    cudaMalloc((void **)&d_ya,  sizeof(float));
	cudaMemcpy(d_xm, xm, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ym, ym, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_xa, xa, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ya, ya, sizeof(float), cudaMemcpyHostToDevice);
	pointcalc<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_n, d_xm, d_ym, d_xa, d_ya);
  	cudaMemcpy(n, d_n, N * sizeof(int), cudaMemcpyDeviceToHost);
  	for (int i = 0; i < N; i++) {
  		no.nx[i] = n[i];
  	} 	
  	nhistory.push_back(no);
	GLfloat* color;
	color = (GLfloat *) malloc (3 * sizeof(GLfloat));
	glBegin(GL_POINTS);
	for(int i=0; i<WINDOW_DIM; i++) {
		for (int j=0; j<WINDOW_DIM; j++) {
			int num = n[i + WINDOW_DIM*j];
			color[0] = colors[num].r;
			color[1] = colors[num].g;
			color[2] = colors[num].b;
			glColor3fv(color);
			glVertex2f(i, j);
		}
	}		
	glEnd();
	glFlush();
	glutSwapBuffers();
}

void mouse(int button, int state, int x, int y) {	
	switch(button) {
		case GLUT_LEFT_BUTTON:
			if(state==GLUT_DOWN) {
				st.x = x; fh.x = x;
				st.y = y; fh.y = y;
				minxs = xmin + (float)x/(WINDOW_DIM)*(xmax - xmin);
				minys = ymin + (float)y/(WINDOW_DIM)*(ymax - ymin);
				click = true;                 
			}
			if(state==GLUT_UP) {
				maxxs = xmin + (float)x/(WINDOW_DIM)*(xmax - xmin);
				maxys = ymin + (float)y/(WINDOW_DIM)*(ymax - ymin);
				xmin = minxs; xmax = maxxs;
				ymin = minys; ymax = maxys;
				if (xmin > xmax) { 
					xmax = xmin + xmax; 
					xmin = xmax - xmin; 
					xmax = xmax - xmin;
					xflag = true; 
				}
				if (ymin > ymax) { 
					ymax = ymin + ymax; 
					ymin = ymax - ymin; 
					ymax = ymax - ymin;
					yflag = true; 
				}
				if (abs(maxys - minys) > abs(maxxs - minxs)) { 
						yflag ? minys = maxys - (maxxs - minxs) : maxys = minys + (maxxs - minxs); 
						yflag ? ymin  = ymax  - (xmax  -  xmin) : ymax  = ymin  + (xmax  -  xmin);
					    yflag = false;
				}
				if (abs(maxxs - minxs) > abs(maxys - minys)) { 
						xflag ? minxs = maxxs - (maxys - minys) : maxxs = minxs + (maxys - minys); 
						xflag ? xmin  = xmax  - (ymax  -  ymin) : xmax  = xmin  + (ymax  -  ymin);
						xflag = false;				
				}
				xminv.push_back(xmin);
				xmaxv.push_back(xmax);
				yminv.push_back(ymin);
				ymaxv.push_back(ymax);
				click = false;
				glutPostRedisplay();
			}
		break;  
	}
}

void motion(int x, int y) {
	GLfloat* w = (GLfloat *)malloc(3*sizeof(GLfloat));
	w[0] = 1; w[1] = 1; w[2] = 1;
	if (!click) return;
	glEnable(GL_COLOR_LOGIC_OP);
	glPolygonMode(GL_FRONT, GL_LINE);
	glDrawBuffer(GL_FRONT);
	tp.x = min((float)fh.x, (float)st.x);
	br.x = max((float)fh.x, (float)st.x);
	tp.y = min((float)fh.y, (float)st.y);
	br.y = max((float)fh.y, (float)st.y);
	if (abs(br.y - tp.y) > abs(br.x - tp.x)) (fh.y < st.y) ? tp.y = br.y  - abs(br.x - tp.x): br.y = tp.y  + abs(br.x - tp.x);
	if (abs(br.x - tp.x) > abs(br.y - tp.y)) (fh.x < st.x) ? tp.x = br.x  - abs(br.y - tp.y): br.x = tp.x  + abs(br.y - tp.y);
	glBegin(GL_QUADS);
	glColor3fv(w);
	glVertex2f(tp.x, tp.y);
	glVertex2f(tp.x, br.y);
	glVertex2f(br.x, br.y);
	glVertex2f(br.x, tp.y);
	tp.x = min((float)x, (float)st.x);
	br.x = max((float)x, (float)st.x);
	tp.y = min((float)y, (float)st.y);
	br.y = max((float)y, (float)st.y);
	fh.x = x;
	fh.y = y;
	if (abs(br.y - tp.y) > abs(br.x - tp.x)) (fh.y < st.y) ? tp.y = br.y  - abs(br.x - tp.x): br.y = tp.y  + abs(br.x - tp.x);
	if (abs(br.x - tp.x) > abs(br.y - tp.y)) (fh.x < st.x) ? tp.x = br.x  - abs(br.y - tp.y): br.x = tp.x  + abs(br.y - tp.y);
	glBegin(GL_QUADS);
	glColor3fv(w);
	glVertex2f(tp.x, tp.y);
	glVertex2f(tp.x, br.y);
	glVertex2f(br.x, br.y);
	glVertex2f(br.x, tp.y);
	glEnd();
	glDisable(GL_COLOR_LOGIC_OP);
	glFlush(); 
	glDrawBuffer(GL_BACK);
}

void keyboard (unsigned char key, int x, int y) {
	if (key == 'b') {
		glMatrixMode(GL_PROJECTION);
    	glLoadIdentity();
    	glOrtho(0, WINDOW_DIM, WINDOW_DIM, 0, 0, 1);
    	glDisable(GL_DEPTH_TEST);
    	glMatrixMode(GL_MODELVIEW);
    	glLoadIdentity();
    	glTranslatef(1, 1, 0);    
    	glClearColor(.0, .0, .0, 0);
    	glClear(GL_COLOR_BUFFER_BIT);
    	struct nh back;
    	if (nhistory.size() >= 2) {
    		nhistory.pop_back();
    		if (xminv.size() >= 2) {
    			xminv.pop_back();
    			xmaxv.pop_back();
    			yminv.pop_back();
    			ymaxv.pop_back();
    		}
    	}
    	if (nhistory.size() >= 1) {
    		xmin = xminv[xminv.size() - 1];
    		xmax = xmaxv[xmaxv.size() - 1];
    		ymin = yminv[yminv.size() - 1];
    		ymax = ymaxv[ymaxv.size() - 1];
    		back = nhistory[nhistory.size() - 1];	
    		GLfloat* color;
        	color = (GLfloat *)malloc(3*sizeof(GLfloat));        
        	glBegin(GL_POINTS);        
        	for(int i=0; i<WINDOW_DIM; i++) {
        		for (int j=0; j<WINDOW_DIM; j++) {
        			int num = back.nx[i + WINDOW_DIM * j];
        			color[0] = colors[num].r;
        			color[1] = colors[num].g;
        			color[2] = colors[num].b;
        			glColor3fv(color);
        			glVertex2f(i, j);
        		}
        	}
    	}	
    	glEnd();    
    	glFlush();
    	glutSwapBuffers();
	}
	if (key == 'q') exit(0);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowPosition(100,100);
	glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
	glutCreateWindow("Mandelbrot Set");
	glEnable(GL_DEPTH_TEST);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	glLogicOp(GL_XOR);
	xminv.push_back(xmin);
    xmaxv.push_back(xmax);
    yminv.push_back(ymin);
    ymaxv.push_back(ymax);
    xm = &xmin;
    ym = &ymin;
    xa = &xmax;
    ya = &ymax;
    n  = (int*)malloc(N * sizeof(int));
  	InitializeColors();
  	glutMainLoop();
  	return 0;
}