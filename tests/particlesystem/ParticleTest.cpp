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

Emitter *fireEmitter1, *fireEmitter2, *debrisEmitter, *smokeEmitter;


void initGL() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glPointSize(3.0);
    shader = new Shader("../src/test/test");
    if (!shader->loaded()) std::cout << shader->errors() << std::endl;
    renderer = new Renderer(sp, shader);
    //smokeTexture = renderer->loadTexture("smoke.png");
    //debrisTexture = renderer->loadTexture("debris.png");
    //blastTexture = renderer->loadTexture("blast.png");

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
        debrisEmitter->burst();
        smokeEmitter->burst();
        fireEmitter1->burst();
        fireEmitter2->burst();
        
        break;
    }
}

void initParticleSystem() {

     GLuint fireTexture1 = renderer->loadTexture("fire1.png");
     GLuint fireTexture2 = renderer->loadTexture("fire2.png");
     GLuint debrisTexture = renderer->loadTexture("debris.png");
     GLuint smokeTexture = renderer->loadTexture("smoke.png");


    // construct particle system
    sp = new ParticleSystem(4);

    // add the emitter
    fireEmitter1 = sp->newEmitter(300);
    fireEmitter1->posIs(Vector3(0.f, 0.5f, 0.f));
    fireEmitter1->burstSizeIs(300);
    fireEmitter1->typeIs(Emitter::EMITTER_BURST);
    fireEmitter1->blendModeIs(Emitter::BLEND_FIRE);
    fireEmitter1->textureIs(fireTexture1);
    fireEmitter1->rateIs(0.02f);
    fireEmitter1->lifeTimeIs(40.f);
    fireEmitter1->massIs(1.f);
    fireEmitter1->posRandWeightIs(0.03);
    fireEmitter1->velIs(Vector3(0.f, 0.f, 0.f));
    fireEmitter1->velRandWeightIs(0.01);
    fireEmitter1->accIs(Vector3(0.f, -0.002f, 0.0f));
    fireEmitter1->pointSizeIs(70.f);
    fireEmitter1->growthFactorIs(0.99f);

    fireEmitter2 = sp->newEmitter(300);
    fireEmitter2->posIs(Vector3(0.f, 0.5f, 0.f));
    fireEmitter2->burstSizeIs(300);
    fireEmitter2->typeIs(Emitter::EMITTER_BURST);
    fireEmitter2->blendModeIs(Emitter::BLEND_FIRE);
    fireEmitter2->textureIs(fireTexture2);
    fireEmitter2->rateIs(0.02f);
    fireEmitter2->lifeTimeIs(40.f);
    fireEmitter2->massIs(1.f);
    fireEmitter2->posRandWeightIs(0.03);
    fireEmitter2->velIs(Vector3(0.f, 0.f, 0.f));
    fireEmitter2->velRandWeightIs(0.01);
    fireEmitter2->accIs(Vector3(0.f, -0.002f, 0.0f));
    fireEmitter2->pointSizeIs(70.f);
    fireEmitter2->growthFactorIs(0.99f);

    debrisEmitter = sp->newEmitter(100);
    debrisEmitter->posIs(Vector3(0.f, 0.5f, 0.f));
    debrisEmitter->burstSizeIs(100);
    debrisEmitter->typeIs(Emitter::EMITTER_BURST);
    debrisEmitter->blendModeIs(Emitter::BLEND_SMOKE);
    debrisEmitter->textureIs(debrisTexture);
    debrisEmitter->rateIs(0.02f);
    debrisEmitter->lifeTimeIs(300.f);
    debrisEmitter->massIs(1.f);
    debrisEmitter->posRandWeightIs(0.02);
    debrisEmitter->velIs(Vector3(0.f, 0.1f, 0.f));
    debrisEmitter->velRandWeightIs(0.02);
    debrisEmitter->accIs(Vector3(0.f, -0.004, 0.0f));
    debrisEmitter->pointSizeIs(10.f);
    debrisEmitter->growthFactorIs(1.f);

    smokeEmitter = sp->newEmitter(5);
    smokeEmitter->posIs(Vector3(0.f, 0.5f, 0.f));
    smokeEmitter->burstSizeIs(5);
    smokeEmitter->typeIs(Emitter::EMITTER_BURST);
    smokeEmitter->blendModeIs(Emitter::BLEND_SMOKE);
    smokeEmitter->textureIs(smokeTexture);
    smokeEmitter->rateIs(0.02f);
    smokeEmitter->lifeTimeIs(70.f);
    smokeEmitter->massIs(1.f);
    smokeEmitter->posRandWeightIs(0.2f);
    smokeEmitter->velIs(Vector3(0.f, 0.001f, 0.f));
    smokeEmitter->velRandWeightIs(0.001);
    smokeEmitter->accIs(Vector3(0.f, 0.0f, 0.0f));
    smokeEmitter->pointSizeIs(100.f);
    smokeEmitter->growthFactorIs(1.02f);





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


