// Draw an Icosahedron
// ECE4893/8893 Project 4
// Seth Stewart

#include <iostream>
#include <math.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <time.h>

using namespace std;

#define NFACE 20
#define NVERTEX 12

#define X .525731112119133606 
#define Z .850650808352039932

// These are the 12 vertices for the icosahedron
static GLfloat vdata[NVERTEX][3] = {    
   {-X, 0.0, Z}, {X, 0.0, Z}, {-X, 0.0, -Z}, {X, 0.0, -Z},    
   {0.0, Z, X}, {0.0, Z, -X}, {0.0, -Z, X}, {0.0, -Z, -X},    
   {Z, X, 0.0}, {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0} 
};

// These are the 20 faces.  Each of the three entries for each 
// vertex gives the 3 vertices that make the face.
static GLint tindices[NFACE][3] = { 
   {0,4,1}, {0,9,4}, {9,5,4}, {4,5,8}, {4,8,1},    
   {8,10,1}, {8,3,10}, {5,3,8}, {5,2,3}, {2,7,3},    
   {7,10,3}, {7,6,10}, {7,11,6}, {11,0,6}, {0,1,6}, 
   {6,1,10}, {9,0,11}, {9,11,2}, {9,2,5}, {7,2,11} };

int testNumber; // Global variable indicating which test number is desired
int w, h, n = 0, depth, iresult, updateRate = 10;
const int maxTriangles = 20480; // 20*(4^5) maximum depth of 5 as specified in project
GLfloat R[maxTriangles];
GLfloat G[maxTriangles];
GLfloat B[maxTriangles];
//GLfloat* colorR;
//GLfloat* colorG;
//GLfloat* colorB;

void drawIcos(int depth);
void drawTriangle(GLfloat* v1, GLfloat* v2, GLfloat* v3);

//20*pow(4^depth)
//for each generate random number and convert to float

void Test1()
{
}

void Test2()
{
}

void Test3()
{
}

void Test4()
{
}

void Test5(int depth)
{
}

void Test6(int depth)
{
}

void init()
{
  glClearColor( 0.0, 0.0, 0.0, 0.0 );
  glShadeModel(GL_FLAT);
  glEnable(GL_LINE_SMOOTH);
}

void normalize(GLfloat v[3])
{
  //calculate the length and divide by the legnth
  GLfloat d = sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
  if (d==0) return;
  v[0] /= d;
  v[1] /= d;
  v[2] /= d;
} 

void rotate()
{
  static GLfloat rotX = 0.0;
  static GLfloat rotY = 0.0;
  glRotatef(rotX, 1.0, 0.0, 0.0);
  glRotatef(rotY, 0.0, 1.0, 0.0);
  rotX += 1.0;
  rotY += 1.0;
}

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  //glColor3f(1.0,1.0,1.0);
  glLoadIdentity();
  gluLookAt( 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  glPushMatrix();
  glTranslatef(w/2, h/2, 0);
  glScalef(w/2,h/2,0);
  if (testNumber == 2 || testNumber == 4 || testNumber == 6){
    rotate(); // rotate 1 degree around the x and y axis
  }
  drawIcos(depth);
  glPopMatrix();
  glutSwapBuffers();
}

void reshape(int w, int h)
{
  glViewport(0.0, 0.0, (GLsizei) w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho((GLdouble) 0.0, (GLdouble)w, (GLdouble) 0.0, (GLdouble) h, (GLdouble) -w, (GLdouble) w);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void timer(int)
{
  glutPostRedisplay();
  glutTimerFunc(1000.0 / updateRate, timer, 0);
}

void drawTriangle(GLfloat* v1, GLfloat* v2, GLfloat* v3, int n)
{ 
  //glColor3f(R[number], G[number], B[number]);//triangle color
  glBegin(GL_TRIANGLES);
  //cout<<"n: "<< n<<endl;
  glColor3f( R[n], G[n], B[n] );
  //glColor3f( 0.0, 0.0, 1.0 );
  // NEEDS TO BE DIFFERENT EACH FACE
  glNormal3fv(v1); glVertex3fv(v1);
  glNormal3fv(v2); glVertex3fv(v2);
  glNormal3fv(v3); glVertex3fv(v3);
  glEnd();
  glLineWidth(2.0);
  glBegin(GL_LINE_LOOP);
  glColor3f(1.0,1.0,1.0); //white lines
  glBegin(GL_LINE_LOOP);
  glNormal3fv(v1); glVertex3fv(v1);
  glNormal3fv(v2); glVertex3fv(v2);
  glNormal3fv(v3); glVertex3fv(v3);
  glEnd();
  
}

void subDivide(GLfloat* v1, GLfloat* v2, GLfloat* v3, int depth)
{
  if(depth == 0)
  {
    drawTriangle(v1,v2,v3,n%iresult);
    n++;
    return;
  }
  GLfloat v12[3];
  GLfloat v23[3];
  GLfloat v31[3];
  for (int i =0; i<3; i ++)
  {
    v12[i] = v1[i] + v2[i];
    v23[i] = v2[i] + v3[i];
    v31[i] = v3[i] + v1[i]; 
  }
  normalize(v12); normalize(v23); normalize(v31);

  subDivide(v1, v12, v31, depth-1);
  subDivide(v2, v23, v12, depth-1);
  subDivide(v3, v31, v23, depth-1);
  subDivide(v12, v23, v31, depth-1);

}

void drawIcos(int depth)
{
  for (int i=0; i<NFACE; i++)
  {
    subDivide(&vdata[tindices[i][0]][0],
	      &vdata[tindices[i][1]][0],
              &vdata[tindices[i][2]][0],
              depth);
  }
}

int main(int argc, char** argv)
{
  srand(time(NULL));
  if (argc < 2)
    {
      std::cout << "Usage: icosahedron testnumber" << endl;
      exit(1);
    }
    // Set the global test number
  testNumber = atoi(argv[1]);
  depth = 0;
  if (testNumber == 3 || testNumber == 4) depth = 1;
  if ( (testNumber == 5 || testNumber ==6) && argc <3)
  {
    cout<<"Depth must be entered for test 5 or 6"<<endl;
    exit(1);
  }
  if (argc > 2 && (testNumber==5 || testNumber ==6))
  {
    depth = atoi(argv[2]);
    if (depth > 5 || depth < 0){ cout<<"Depth must be between 0 and 5"<<endl; exit(1); }
  }
 
  //Generate color values
  for (int i=0; i< (int)(20*pow(4,depth)); i++)
  {
    R[i] = ((double)rand() / (RAND_MAX));
    G[i] = ((double)rand() / (RAND_MAX));
    B[i] = ((double)rand() / (RAND_MAX));    
  }
  h = 500;
  w = 500;
  iresult = 20*pow(4, depth);

  // Initialize glut  and create your window here
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  glutInitWindowSize(w, h);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("Project 4 Icosahedron");
  init();
  // Set your glut callbacks here
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutTimerFunc(1000.0 / updateRate, timer, 0);
  // Enter the glut main loop here
  glutMainLoop();
  return 0;
}

