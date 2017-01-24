// Draw simple circle
// George F. Riley, Georgia Tech, Fall  2011

#include <iostream>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <map>

#include <GL/glut.h>
#include <GL/glext.h>
#include <GL/gl.h>
#include <GL/glu.h>

using namespace std;

static const float DEG2RAD = M_PI/180;
static int updateRate = 50;

void drawCircle()
{
  // Draw a circle by drawing 360 small lines
  GLfloat radius = 1.0;
  glColor3f(1.0, 1.0, 1.0);
  glLineWidth(2.0);
  glBegin(GL_LINE_LOOP);
  for (int i = 0; i < 360; i++)
    {
      float degInRad = i*DEG2RAD;
      glVertex2f(cos(degInRad)*radius,sin(degInRad)*radius);
   }
  glEnd();
  // Add spokes each 30 degrees
  for (int i = 0; i < 360; i += 30)
    {
      // Just for fun make spokes different colors
      glColor3f(i / 360.0, 360.0 - i / 360.0, 0.5);
      float degInRad = i*DEG2RAD;
      glBegin(GL_LINES);
      glVertex2f(0.0, 0.0);
      glVertex2f(cos(degInRad), sin(degInRad));
      glEnd();
    }
}

void display(void)
{
  static int pass;
  
  cout << "Displaying pass " << ++pass << endl;
  // clear all
  glClear(GL_COLOR_BUFFER_BIT);
  // Clear the matrix
  glLoadIdentity();
  // Set the viewing transformation
  gluLookAt(0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0); // try frustrum
  //gluLookAt(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  // Add some slight animation
  static GLfloat transX = 0.0;
  static GLfloat transY = 0.0;
  static GLfloat transZ = 0.0;
  static bool    adding = true;
  glTranslatef(transX, transY, transZ);
  //glTranslatef(transX, transY, 0.0);
  if (adding)
    {
      transX += 1.0;
      transY += 1.0;
      transZ += 1.0;
      if (transX > 400) adding = false;
    }
  else
    {
      transX -= 1.0;
      transY -= 1.0;
      transZ -= 1.0;
      if (transX < 0.0) adding = true;
    }
  //glScalef(1000.0, 1000.0, 0);
  glScalef(100.0, 100.0, 0);
  //glScalef(1.0, 1.0, 0);
  // Try rotating
  static GLfloat rotX = 0.0;
  static GLfloat rotY = 0.0;
  glRotatef(rotX, 1.0, 0.0, 0.0);
  glRotatef(rotY, 0.0, 1.0, 0.0);
  rotX += 1.0;
  rotY -= 1.0;
  drawCircle();
  // Flush buffer
  //glFlush(); // If single buffering
  glutSwapBuffers(); // If double buffering
}

void init()
{
  //select clearing (background) color
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glShadeModel(GL_FLAT);
}

void reshape(int w, int h)
{
  glViewport(0,0, (GLsizei)w, (GLsizei)h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, (GLdouble)w, (GLdouble)0.0, h, (GLdouble)w, (GLdouble)-w);
  //glFrustum(0.0, (GLdouble)w, (GLdouble)0.0, h, (GLdouble)1, (GLdouble)40);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void timer(int)
{
  glutPostRedisplay();
  glutTimerFunc(1000.0 / updateRate, timer, 0);
}

int main(int argc, char** argv)
{
  if (argc > 1) updateRate = atof(argv[1]);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowSize(500, 500);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("Circle");
  init();
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutTimerFunc(1000.0 / updateRate, timer, 0);
  glutMainLoop();
  return 0;
}

