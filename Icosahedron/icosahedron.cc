// Draw an Icosahedron
// ECE4893/8893 Project 4
// Seth Stewart

#include <iostream>
#include <math.h>
#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/gl.h>
#include <GL/glu.h>

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
int w,h, depth, updateRate=100; 
void drawIcos(int depth);

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

void display(void)
{
  glClear(GL_COLOR_BUFFER_BIT);
  glClear(GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glColor3f(1.0,1.0,1.0);
  glLoadIdentity();
  gluLookAt( 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  glPushMatrix();
  glTranslatef(w/2, h/2, 0);
  glScalef(w/2,h/2,0);
  if (testNumber == 2 || testNumber == 4 || testNumber == 6){
    glRotatef ( 1.0, 1.0, 1.0, 0.0); // rotate 1 degree around the x and y axis
    //fix 
  }
  //drawIcos(depth);
  //glPopMatrix();
  drawIcos(depth);
  glutSwapBuffers();
}

void reshape(int w, int h)
{
  glViewport(0.0, 0.0, (GLsizei) w, (GLsizei)h);
  glLoadIdentity();
  glOrtho( 0, w, 0, h, -w, w);
  glMatrixMode(GL_MODELVIEW);
}

void timer(int)
{
  glutTimerFunc(1000.0 / updateRate, timer, 0);
}

void drawTriangle(GLfloat* v1, GLfloat* v2, GLfloat* v3)
{
  glBegin(GL_TRIANGLES);
  glColor3f(0.0,0.0,1.0);//triangle color
  // NEEDS TO BE DIFFERENT EACH FACE
  glNormal3fv(v1); glVertex3fv(v1);
  glNormal3fv(v2); glVertex3fv(v2);
  glNormal3fv(v3); glVertex3fv(v3);
  glEnd();
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
    drawTriangle(v1,v2,v3);
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
  if (argc < 2)
    {
      std::cout << "Usage: icosahedron testnumber" << endl;
      exit(1);
    }
    // Set the global test number
  testNumber = atoi(argv[1]);
  cout<<"testNumber = "<<testNumber<<endl;
  if (argc > 2)
  { 
    depth = atoi(argv[2]); 
  }else
  { 
    depth = 1;
  }
  cout<<"depth = "<<depth<<endl; 
  int h = 500;
  int w = 500;
  //cout<< "Test"<<endl;
  // Initialize glut  and create your window here
  glutInit(&argc, argv);
  glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH );
  //cout<< "Test"<<endl;
  glutInitWindowSize(w, h);
  glutInitWindowPosition(100, 100);
  //glutInitWindowSize(w, h);
  glutCreateWindow("Project 4 Icosahedron");
  //glutCreateWindow("Project 4 Icosahedron");
  init();
  //glutInitWindowPosition(100, 100);
 // cout<< "Test"<<endl;
  //glutCreateWindow("Project 4 Icosahedron");
  // Set your glut callbacks here
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  //glutTimerFunc(1000.0 / updateRate, timer, 0);
  // Enter the glut main loop here
  glutMainLoop();
  return 0;
}

