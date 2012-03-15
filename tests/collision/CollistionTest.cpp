#include "../../src/MathEngine.h"
#include "../../src/HitBox.h"
#include "../../src/Projectile.h"
#include "../../src/Target.h"
#include "../../src/Weapon.h"

#include <vector>
#include <string>
#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>

int width = 800;
int height = 600;

std::vector<Weapon> weapons;
std::vector<Target> targets;

void initGL() {
    glEnable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    gluPerspective(60.f, width/height, 0.1f, 1000.f);
    glMatrixMode(GL_MODELVIEW);
}

void init() {
    weapons.push_back( Weapon("gun", Vector3(0.f, 0.f, 0.f), 5.f, 10.f, 30.f) );
    targets.push_back( Target("boat", 
                               Vector3(10.f, 0.f, 10.f),
                               Vector3(8.f, -2.f, 8.f),
                               Vector3(12.f, 2.f, 12.f),
                               30.f) );
}

void idle() {
   for (unsigned int i=0; i<weapons.size(); i++) {
        weapons.at(i).updateProjectiles(0.01f);
        for (unsigned int j=0; j<weapons.at(i).projectile.size(); j++) {
            Projectile p = weapons.at(i).projectile.at(j);
            if (p.hitBoxTest(&targets.at(0).hitBox)) {
                std::cout << "HIT" << std::endl;
            }
        }
    }
    glutPostRedisplay();
}

void fireWeapon() {
    if (!weapons.empty()) {
        weapons.at(0).fire(Vector3(1.f, 0.f, 1.f));
    }
}


void display() {
    glClearColor(0.8f, 0.8f, 0.8f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glLoadIdentity();

    gluLookAt(-4.f, 3.f, -2.f, 10.f, 0.f, 10.f, 0.f, 1.f, 0.f);

    glColor3f(0.f, 0.f, 0.f);
    glutWireCube(0.2f);

    // render the targets' hitboxes
    for (unsigned int i=0; i<targets.size(); i++) {
        Vector3 p0 = targets.at(i).hitBox.p0;
        Vector3 p1 = targets.at(i).hitBox.p1;
        glColor3f(0.f, 0.f, 0.f);
        glBegin(GL_LINES);
            glVertex3f(p0.x, p0.y, p0.z);
            glVertex3f(p1.x, p0.y, p0.z);
            glVertex3f(p0.x, p0.y, p0.z);
            glVertex3f(p0.x, p1.y, p0.z);
            glVertex3f(p0.x, p0.y, p0.z);
            glVertex3f(p0.x, p0.y, p1.z);
            glVertex3f(p1.x, p1.y, p1.z);
            glVertex3f(p0.x, p1.y, p1.z);
            glVertex3f(p1.x, p1.y, p1.z);
            glVertex3f(p1.x, p0.y, p1.z);
            glVertex3f(p1.x, p1.y, p1.z);
            glVertex3f(p1.x, p1.y, p0.z);
        glEnd();
    }

    // render projectiles
    for (unsigned int i=0; i<weapons.size(); i++) {
        Weapon w = weapons.at(i);
        for (unsigned int j=0; j<w.projectile.size(); ++j) {
            Projectile p = w.projectile.at(j);
            glPushMatrix();
            glTranslatef(p.position.x, p.position.y, p.position.z);
            glutSolidSphere(0.2, 4, 4);
            glPopMatrix();
        }
    }

    glutPostRedisplay();
    glutSwapBuffers();
}

void reshape(int x, int y) {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, x, y);
    gluPerspective(60.f, width/height, 0.1f, 1000.f);
    glMatrixMode(GL_MODELVIEW);
}

void keyPressed (unsigned char key, int x, int y) {
    switch (key) {
    case 'f':
        fireWeapon();
        break;
    }
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(width, height);
    glutCreateWindow("Collision test");
    glewInit();
    glutReshapeFunc(reshape);
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyPressed);
    init();
    glutMainLoop();
    return 0;
}







