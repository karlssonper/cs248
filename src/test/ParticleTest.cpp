#include "../ParticleSystem.h"

#include "Renderer.h"
#include "Shader.h"

#include <GL/glew.h>
#include <GL/glut.h>

float width = 500;
float height = 500;
float fov = 50.0;

ParticleSystem *sp;
Renderer *renderer;
Shader *shader;

void initGL() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glPointSize(3.0);
    shader = new Shader("../../src/test");
    renderer = new Renderer(sp->emitter(0)->vboPos(), shader);
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glDepthMask(GL_TRUE);
}

void idle() {
    sp->update(0.002);
    glutPostRedisplay();
}

void display() {
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
     glMatrixMode(GL_MODELVIEW);
     glLoadIdentity();
     glColor4f(0.0f, 0.0f, 1.0f, 1.0f);

     renderer->render();

     glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
     glutWireCube(2.0);
     glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, (float)w/(float)h, 0.1f, 100.0);
    gluLookAt(0.0, 0.5, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void keyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
    case 'q':
    case 'Q':
        exit(0);
        break;
    }
}

void initParticleSystem() {
     // construct emitter
    Emitter::EmitterParams params;
    params.numParticles_ = 50;
    params.mass_ = 1.f;
    params.rate_ = 3.f;
    params.startAcc_[0] = 0.f;
    params.startAcc_[1] = -1.f;
    params.startAcc_[2] = 0.f;
    params.startVel_[0] = 1.f;
    params.startVel_[1] = 1.f;
    params.startVel_[2] = 1.f;
    params.startPos_[0] = 1.f;
    params.startPos_[1] = 1.f;
    params.startPos_[2] = 1.f;
    params.color_[0] = 1.f;
    params.color_[1] = 1.f;
    params.color_[2] = 1.f;

    // construct particle system
    sp = new ParticleSystem(1);

    // add the emitter
    sp->newEmitter(params);
}

void cleanUp() {
    delete sp;
}

int main(int argc, char** argv) {

   

    // glut and glew 
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(width, height);
    glutCreateWindow("Particle test");
    glewInit();
    initParticleSystem();
    initGL();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);

    // run!
    glutMainLoop();

   
    return 0;
}


