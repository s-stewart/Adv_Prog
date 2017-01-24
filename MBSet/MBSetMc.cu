// Calculate and Display the Mandlebrot Fractal
// using the NVIDIA CUDA framework
//
// ECE 4893 Fall 2012
// Mac Clayton
// 28 October, 2012

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <GL/freeglut.h>
#include <sys/time.h>
#include "Complex.cu"
// #include "cuPrintf.cu"   // Debugging tool for printing from kernel

// Size of window in pixels, both width and height
#define WINDOW_DIM            (512)
// Maximum iterations for the MBSet calculations
#define MAX_IT                (2000)

using namespace std;

// RGB Structure for Color manipulation
struct RGB
{
    float r, g, b;
};
// Point structure for holding box points
struct Point
{
    int x, y;
};

// Initial screen coordinates, both host and device.
Complex minC(-2.0f, -1.2f);
Complex maxC(1.0f, 1.8f);
Complex* d_minC;
Complex* d_maxC;

// Timing variables
struct timeval t1;
struct timeval t2;
double seconds;

int *array;                                         // Host array for Mandlebrot
int *d_array;                                       // Device array for Mandlebrot
int size = WINDOW_DIM * WINDOW_DIM * sizeof(int);   // Size of Array
int sizeCpx = sizeof(Complex);                      // Size of Complex Class

// Create a stack to keep track of zooming:
stack< pair< Complex, Complex > > mouseBox; 

int mousePosX = 0;      // Starting X position
int mousePosY = 0;      // Starting Y position

Point box1;
Point box2;

bool isRectDrawn;

// Global Variables for color array
float colorR[MAX_IT];
float colorG[MAX_IT];
float colorB[MAX_IT];

// Block and thread counts for CUDA
dim3 threads(8, 8);
dim3 blocks(WINDOW_DIM/threads.x,
            WINDOW_DIM/threads.y);


// Converts hue (0 - 360) to RGB value
// (Assumes Saturation = 1 and Value = 1)
RGB hueToRGB(float h)
{
    RGB color;
    float r, g, b;
    int i = (int)floor(h/60.0f);
    float f = h/60.0f - i;
    float p = 0;
    float q = (1 - f);
    float t = (1 - (1 - f));
    switch(i % 6)
    {
        case 0:
        {
            r = 1;
            g = t;
            b = p;
            break;
        }
        case 1:
        {
            r = q;
            g = 1;
            b = p;
            break;
        }
        case 2:
        {
            r = p;
            g = 1;
            b = p;
            break;
        }
        case 3:
        {
            r = p;
            g = q;
            b = 1;
            break;
        }
        case 4:
        {
            r = t;
            g = p;
            b = 1;
            break;
        }
        case 5:
        {
            r = 1;
            g = p;
            b = q;
            break;
        }
    }
    color.r = r;
    color.g = g;
    color.b = b;
    return color;
}


// Interpolate and populate
void generateColorArrays(int iterations)
{
    RGB temp;
    float x;
    for(int i = 0; i < iterations; ++i)
    {
        x = (i % 36) * 10;
        temp = hueToRGB(x);
        colorR[i] = temp.r;
        colorG[i] = temp.g;
        colorB[i] = temp.b;
    }
}


// CUDA thread to calculate Mandlebrot pixel
__global__ void mbCalc(Complex* d_minC, Complex* d_maxC, int* array)
{
    // Calculate Thread Index and Pixel location
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    // int index threadIdx.x + blockIdx.x * blockDim.x;
    int index = x + y * blockDim.x * gridDim.x;

    // Interpolate:
    float xVal = d_minC->r + x * (d_maxC->r - d_minC->r) / (WINDOW_DIM - 1);
    float yVal = d_minC->i + y * (d_maxC->i - d_minC->i) / (WINDOW_DIM - 1);

    // Setup c and Z
    Complex c = Complex(xVal, yVal);
    Complex Z = Complex(c);

    int i = 0;
    for(; i < MAX_IT; ++i)
    {
        Z = Z * Z + c;
        if(Z.magnitude2() > 4.0f) break;
    }    
    // Store the value of i into the array:
    array[index] = i;
}

// Setup the memory and call the kernel
__host__ void getMBData()
{
    // Copy Data to Device
    cudaMemcpy(d_minC, &minC, sizeCpx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxC, &maxC, sizeCpx, cudaMemcpyHostToDevice);

    // Run threads:
    gettimeofday(&t1,0);    // Timing
    mbCalc<<<blocks, threads>>>(d_minC, d_maxC, d_array);
    // Block until all threads are finished
    cudaDeviceSynchronize();
    gettimeofday(&t2,0);    // Timing

    // Timing Output
    seconds = (t2.tv_sec-t1.tv_sec)*100000 + t2.tv_usec - t1.tv_usec;
    cout << "Calculated in " << seconds/1000.0 << " milliseconds" << endl;
    
    // Copy Data Back:
    cudaMemcpy(array, d_array, size, cudaMemcpyDeviceToHost);
}

// Debug code to print out MBSet integers
__host__ void printMBset(int* array)
{
	cout << endl;
	for(int i = 1; i < (WINDOW_DIM*WINDOW_DIM + 1); ++i)
	{
		cout << array[i-1] << " ";
		if(((i % (WINDOW_DIM)) == 0) && (i > 1))
		{
			cout << endl;
		}
	}
	cout << endl;
}

// OpenGL reshape function
__host__ void reshape(int w, int h)
{
    glViewport(0,0, (GLsizei)w, (GLsizei)h);    // Setup Viewport Size
    glMatrixMode(GL_PROJECTION);                // Switch to Projection Matrix
    glLoadIdentity();                           // Reset projection matrix to identity
    glOrtho(0.0, w, 0.0, h, -10, 10);           // Left, Right, Bottom, Top
    glMatrixMode(GL_MODELVIEW);                 // Load Modelview Matrix
    glLoadIdentity();                           // reset modelview matrix to identity
    glutPostRedisplay();                        // Mark display for redrawing
}

// OpenGL display function
__host__ void display(void)
{ 
    int count;
    glClear(GL_COLOR_BUFFER_BIT);               // Clear the Color Buffer
  
    // draw the set
    glBegin(GL_POINTS);
    for(int x = 0; x < WINDOW_DIM; ++x)
    {
        for (int y = 0; y < WINDOW_DIM; ++y)
        {
            count = array[y * WINDOW_DIM + x];  // Get data from array
            if(count == MAX_IT)                 // If the pixel is black
            {
                glColor3f(0.0, 0.0, 0.0);
                glVertex2f(x,y);
            }
            else if(count > 0)                  // If the pixel is colored
            {
                glColor3f(colorR[count], colorG[count] , colorB[count]);
                glVertex2f(x,y);
            }
        }
    }
    glEnd();

    // Draw red rectangle
    if(isRectDrawn)
    {
        glColor3f(1.0, 0.0, 0.0);   // Red Line
        glLineWidth(2.0f);          // Draw a thicker line
        glBegin(GL_LINE_LOOP);      // Connect the dots
        glVertex2f(box1.x, box1.y);
        glVertex2f(box2.x, box1.y);
        glVertex2f(box2.x, box2.y);
        glVertex2f(box1.x, box2.y);
        glEnd();
    }
    glutSwapBuffers();              // Swap to displayed pixels
}


// OpenGL function for mouse interaction
__host__ void mouse(int button, int state, int x, int y)
{ 
    float temp;
    if(button == GLUT_LEFT_BUTTON)
    {
        if (state == GLUT_DOWN)
        {
            box1.x = x;
            box1.y = WINDOW_DIM - y;      
            isRectDrawn = true;
        }
        else
        {
            isRectDrawn = false;
            mouseBox.push(pair<Complex,Complex>(minC, maxC));
            // Make the box square:
            if(box1.x > box2.x)
            {
                temp = box2.x;      // Swap values
                box2.x = box1.x;
                box1.x = temp;
            }
            if(box1.y > box2.y)
            {
                temp = box2.y;      // Swap values
                box2.y = box1.y;
                box1.y = temp;
            }

            // Calculate new minC and maxC
            minC = Complex(minC.r + ((float) box1.x) / WINDOW_DIM * (maxC.r - minC.r),  // Real 
                           minC.i + ((float) box1.y) / WINDOW_DIM * (maxC.i - minC.i)); // Imag
            maxC = Complex(minC.r + ((float) box2.x) / WINDOW_DIM * (maxC.r - minC.r),  // Real
                           minC.i + ((float) box2.y) / WINDOW_DIM * (maxC.i - minC.i)); // Imag
            // Get the Mandlebrot Data
            getMBData();
            // Mark window to redraw
            glutPostRedisplay();
        }
    }
}

// Mouse motion
__host__ void motion(int x, int y)
{ 
    // Rectangle bounds
    int dx, dy;
    box2.x = x;
    box2.y = WINDOW_DIM - y;

    // Make box square
    dx = box2.x - box1.x;
    if (dx < 0) dx *= -1;
    dy = box2.y - box1.y;
    if (dy < 0) dy *= -1;
    if (dx > dy)
    {
        if (box2.y > box1.y) box2.y = box1.y + dx;
        else box2.y = box1.y - dx;
    }
    else
    {
        if (box2.x > box1.x) box2.x = box1.x + dy;
        else box2.x = box1.x - dy;
    }
    glutPostRedisplay();
}

// Keyboard Processing
__host__ void keyboard(unsigned char c, int x, int y)
{
    // Reverse Mouse zoom "b" for back
    if ( c == 'b')
    {
        if(mouseBox.empty())
        {
        cout << "Reached Top Level" << endl;
        }
        else
        {
            cout << "Zooming Out" << endl;
            minC = mouseBox.top().first;
            maxC = mouseBox.top().second;
            mouseBox.pop();                         // Pop off the stack
            getMBData();                            // Get MB Data
            glutPostRedisplay();                    // Mark window to be redisplayed
        }
    }
}

// Not sure if this is needed,
// but what the heck...
__host__ void cleanUP()
{
    free(array);
    cudaFree(d_array);
    cudaFree(d_minC);
    cudaFree(d_maxC);
    cout << "Exited Cleanly!" << endl;
}

// Main!
int main(int argc, char** argv)
{
    // Call cleanUP on Exit
    atexit(cleanUP);
    // Allocate space for Host array:
    array = (int*)malloc(size);
    if(cudaSuccess != cudaMalloc((void **) &d_array, size))   printf("Error1 \n");
    if(cudaSuccess != cudaMalloc((void **) &d_minC, sizeCpx)) printf("Error2 \n");
    if(cudaSuccess != cudaMalloc((void **) &d_maxC, sizeCpx)) printf("Error3 \n");

    generateColorArrays(MAX_IT);                    // Generate RGB colors
    getMBData();                                    // Get initial data

    glutInit(&argc, argv);                          // Init glut
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);    // Double Buffering
    glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);     // Set Window Size
    glutInitWindowPosition(20,20);                  // Set starting location
    glutCreateWindow("MBSet");                      // Create Window
    //glClearColor(1.0, 1.0, 1.0, 0.0);               // Set background to white
    glClearColor(1.0, 0, 0, 0);
    glutDisplayFunc(display);                       // Set display function
    glutReshapeFunc(reshape);                       // Set reshape function
    glutMouseFunc(mouse);                           // Set mouse function
    glutMotionFunc(motion);                         // set motion function
    glutKeyboardFunc(keyboard);                     // set keyboard function

    glutMainLoop();
    return 0;
}