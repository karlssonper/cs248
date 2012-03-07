#include "../ParticleSystem.h"
#include "../MathEngine.h"

#include "Renderer.h"
#include "Shader.h"
#include <iostream>

#include <GL/glew.h>
#include <GL/glut.h>

float width = 500;
float height = 500;
float fov = 50.0;

ParticleSystem *sp;
Renderer *renderer;
Shader *shader;

Vector3 emitterPos0, emitterPos1, emitterPos2;

void initGL() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glPointSize(3.0);
    shader = new Shader("../src/test/test");
    if (!shader->loaded()) std::cout << shader->errors() << std::endl;
    renderer = new Renderer(sp, shader);
    renderer->loadTexture("sprite.png");
    glEnable(GL_POINT_SPRITE);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
    glDepthMask(GL_TRUE);
}

void idle() {
    sp->update(1.f);
    glutPostRedisplay();
}

void display() {
     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
     glMatrixMode(GL_MODELVIEW);
     glLoadIdentity();
     glColor4f(0.0f, 0.0f, 1.0f, 1.0f);

     renderer->render();


     glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
     //glutWireCube(2.0);
     glutSwapBuffers();
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(fov, (float)w/(float)h, 0.1f, 100.0);
    gluLookAt(0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
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
    case 'e':
    case 'E':
        sp->emitter(2)->burst();
        break;
    }
}

void initParticleSystem() {

    emitterPos0 = Vector3(-1.5f, -0.5f, 0.f);
    emitterPos1 = Vector3(1.5f, -0.5f, 0.f);
    emitterPos2 = Vector3(0.0, -0.5f, 0.f);
   

    // construct particle system
    sp = new ParticleSystem(3);

    // add the emitter
    sp->newEmitter(10000);
    sp->emitter(0)->typeIs(Emitter::EMITTER_STREAM);
    sp->emitter(0)->rateIs(0.1f);
    sp->emitter(0)->lifeTimeIs(60.f);
    sp->emitter(0)->massIs(1.f);
    sp->emitter(0)->posIs(emitterPos0);
    sp->emitter(0)->posRandWeightIs(0.05);
    sp->emitter(0)->velIs(Vector3(0.f, 0.f, 0.f));
    sp->emitter(0)->velRandWeightIs(0.01);
    sp->emitter(0)->accIs(Vector3(0.f, -0.01f, 0.0f));
    sp->emitter(0)->colIs(Vector3(1.f, 1.f, 0.f));

    sp->newEmitter(10000);
    sp->emitter(1)->typeIs(Emitter::EMITTER_STREAM);
    sp->emitter(1)->rateIs(0.2f);
    sp->emitter(1)->lifeTimeIs(70.f);
    sp->emitter(1)->massIs(1.f);
    sp->emitter(1)->posIs(emitterPos1);
    sp->emitter(1)->posRandWeightIs(0.01);
    sp->emitter(1)->velIs(Vector3(0.f, 0.2f, 0.f));
    sp->emitter(1)->velRandWeightIs(0.007);
    sp->emitter(1)->accIs(Vector3(0.f, -0.01f, 0.0f));
    sp->emitter(1)->colIs(Vector3(1.f, 0.f, 0.f));

    sp->newEmitter(10000);
    sp->emitter(2)->typeIs(Emitter::EMITTER_BURST);
    sp->emitter(2)->burstSizeIs(300);
    sp->emitter(2)->lifeTimeIs(70.f);
    sp->emitter(2)->massIs(1.f);
    sp->emitter(2)->posIs(emitterPos2);
    sp->emitter(2)->posRandWeightIs(0.01);
    sp->emitter(2)->velIs(Vector3(0.f, 0.0f, 0.f));
    sp->emitter(2)->velRandWeightIs(0.03);
    sp->emitter(2)->accIs(Vector3(0.f, -0.01f, 0.0f));
    sp->emitter(2)->colIs(Vector3(1.f, 0.f, 0.f));


}

void cleanUp() {
    delete sp;
}

int main(int argc, char** argv) {
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
    glutMainLoop();
    return 0;
}


