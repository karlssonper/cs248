#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glut.h>
#include <ParticleSystem.h>
#include <MathEngine.h>
#include <cuda/Emitter.cuh>
#include <iostream>
#include <Camera.h>
#include <Mesh.h>
#include <Node.h>
#include <wrappers/Assimp2Mesh.h>

float width = 500;
float height = 500;
float fov = 50.0;

ParticleSystem *sp;

Mesh * mesh;
ShaderData * shader;
Node * node;
//Renderer *renderer;
//Shader *shader;

Emitter *fireEmitter1, *fireEmitter2, *debrisEmitter, *smokeEmitter;
ShaderData *fireEmitter1sd, *fireEmitter2sd, *debrisEmittersd, *smokeEmittersd;
int mouseX, mouseY;

void initGL() {
    Camera::instance().projectionIs(fov, width/height, 1.f, 100.f);
    /*glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    shader = new Shader("../src/test/test");
    if (!shader->loaded()) std::cout << shader->errors() << std::endl;
    renderer = new Renderer(sp, shader);
    //smokeTexture = renderer->loadTexture("smoke.png");
    //debrisTexture = renderer->loadTexture("debris.png");
    //blastTexture = renderer->loadTexture("blast.png");


    glDepthMask(GL_TRUE);
    glPointSize(3.0);
glEnable(GL_VERTEX_PROGRAM_POINT_SIZE_NV);
        glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
        */

     
}

void idle() {
    sp->update(1.f);
    glutPostRedisplay();
}

void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera::instance().BuildViewMatrix();
    mesh->display();
    sp->display();
     /*glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 
     glMatrixMode(GL_MODELVIEW);
     glLoadIdentity();
     glColor4f(0.0f, 0.0f, 1.0f, 1.0f);

     renderer->render();


     glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
     //glutWireCube(2.0);*/
     glutSwapBuffers();
}

static void keyboard(unsigned char key, int x, int y) {
    switch (key){
        case 27:
            exit(0);
        case 'w':
            Camera::instance().move(0.5);
            break;
        case 's':
            Camera::instance().move(-0.5);
            break;
        case 'a':
            Camera::instance().strafe(-0.5);
            break;
        case 'd':
            Camera::instance().strafe(0.5);
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
void mouseFunc(int x,int y)
{
    int dx = x - mouseX;
    int dy = y - mouseY;
    mouseX = x;
    mouseY = y;
    Camera::instance().yaw(1.6*dx);
    Camera::instance().pitch(1.6*dy);
}
void mouseMoveFunc(int x,int y)
{
    mouseX = x;
    mouseY = y;
}

void reshape(int x, int y)
{

}

void initParticleSystem() {
    std::string s1("../shaders/particle");
    std::string s2("../shaders/particle");
    std::string s3("../shaders/particle");
    std::string s4("../shaders/particle");

    fireEmitter1sd = new ShaderData(s1);
    fireEmitter2sd = new ShaderData(s2);
    debrisEmittersd = new ShaderData(s3);
    smokeEmittersd = new ShaderData(s4);

    std::string t1("sprite");
    std::string t2("sprite");
    std::string t3("sprite");
    std::string t4("sprite");

    std::string p1("../textures/fire1.png");
    std::string p2("../textures/fire2.png");
    std::string p3("../textures/debris.png");
    std::string p4("../textures/smoke.png");

    fireEmitter1sd->enableMatrix(MODELVIEW);
    fireEmitter2sd->enableMatrix(MODELVIEW);
    debrisEmittersd->enableMatrix(MODELVIEW);
    smokeEmittersd->enableMatrix(MODELVIEW);

    fireEmitter1sd->enableMatrix(PROJECTION);
    fireEmitter2sd->enableMatrix(PROJECTION);
    debrisEmittersd->enableMatrix(PROJECTION);
    smokeEmittersd->enableMatrix(PROJECTION);

    fireEmitter1sd->addTexture(t1,p1);
    fireEmitter2sd->addTexture(t2,p2);
    debrisEmittersd->addTexture(t3,p3);
    smokeEmittersd->addTexture(t4,p4);
     //GLuint fireTexture1 = renderer->loadTexture("fire1.png");
     //GLuint fireTexture2 = renderer->loadTexture("fire2.png");
     //GLuint debrisTexture = renderer->loadTexture("debris.png");
     //GLuint smokeTexture = renderer->loadTexture("smoke.png");


    // construct particle system
    sp = new ParticleSystem(4);

    // add the emitter
    fireEmitter1 = sp->newEmitter(300,fireEmitter1sd);
    fireEmitter1->posIs(Vector3(0.f, 0.5f, 0.f));
    fireEmitter1->burstSizeIs(300);
    fireEmitter1->typeIs(Emitter::EMITTER_BURST);
    fireEmitter1->blendModeIs(Emitter::BLEND_FIRE);
    //fireEmitter1->shaderDataIs(fireEmitter1sd);
    //fireEmitter1->textureIs(fireTexture1);
    fireEmitter1->rateIs(0.02f);
    fireEmitter1->lifeTimeIs(40.f);
    fireEmitter1->massIs(1.f);
    fireEmitter1->posRandWeightIs(0.03);
    fireEmitter1->velIs(Vector3(0.f, 0.f, 0.f));
    fireEmitter1->velRandWeightIs(0.01);
    fireEmitter1->accIs(Vector3(0.f, -0.002f, 0.0f));
    fireEmitter1->pointSizeIs(70.f);
    fireEmitter1->growthFactorIs(0.99f);

    fireEmitter2 = sp->newEmitter(300,fireEmitter2sd);
    fireEmitter2->posIs(Vector3(0.f, 0.5f, 0.f));
    fireEmitter2->burstSizeIs(300);
    fireEmitter2->typeIs(Emitter::EMITTER_BURST);
    fireEmitter2->blendModeIs(Emitter::BLEND_FIRE);
    //fireEmitter2->shaderDataIs(fireEmitter2sd);
    //fireEmitter2->textureIs(fireTexture2);
    fireEmitter2->rateIs(0.02f);
    fireEmitter2->lifeTimeIs(40.f);
    fireEmitter2->massIs(1.f);
    fireEmitter2->posRandWeightIs(0.03);
    fireEmitter2->velIs(Vector3(0.f, 0.f, 0.f));
    fireEmitter2->velRandWeightIs(0.01);
    fireEmitter2->accIs(Vector3(0.f, -0.002f, 0.0f));
    fireEmitter2->pointSizeIs(70.f);
    fireEmitter2->growthFactorIs(0.99f);

    debrisEmitter = sp->newEmitter(100,debrisEmittersd);
    debrisEmitter->posIs(Vector3(0.f, 0.5f, 0.f));
    debrisEmitter->burstSizeIs(100);
    debrisEmitter->typeIs(Emitter::EMITTER_BURST);
    debrisEmitter->blendModeIs(Emitter::BLEND_SMOKE);
    //debrisEmitter->shaderDataIs(debrisEmittersd);
    //debrisEmitter->textureIs(debrisTexture);
    debrisEmitter->rateIs(0.02f);
    debrisEmitter->lifeTimeIs(300.f);
    debrisEmitter->massIs(1.f);
    debrisEmitter->posRandWeightIs(0.02);
    debrisEmitter->velIs(Vector3(0.f, 0.1f, 0.f));
    debrisEmitter->velRandWeightIs(0.02);
    debrisEmitter->accIs(Vector3(0.f, -0.004, 0.0f));
    debrisEmitter->pointSizeIs(10.f);
    debrisEmitter->growthFactorIs(1.f);

    smokeEmitter = sp->newEmitter(5,smokeEmittersd);
    smokeEmitter->posIs(Vector3(0.f, 0.5f, 0.f));
    smokeEmitter->burstSizeIs(5);
    smokeEmitter->typeIs(Emitter::EMITTER_BURST);
    smokeEmitter->blendModeIs(Emitter::BLEND_SMOKE);
    //smokeEmitter->shaderDataIs(smokeEmittersd);
    //smokeEmitter->textureIs(smokeTexture);
    smokeEmitter->rateIs(0.02f);
    smokeEmitter->lifeTimeIs(70.f);
    smokeEmitter->massIs(1.f);
    smokeEmitter->posRandWeightIs(0.2f);
    smokeEmitter->velIs(Vector3(0.f, 0.001f, 0.f));
    smokeEmitter->velRandWeightIs(0.001);
    smokeEmitter->accIs(Vector3(0.f, 0.0f, 0.0f));
    smokeEmitter->pointSizeIs(100.f);
    smokeEmitter->growthFactorIs(1.02f);

    node = new Node("sixtenNode");
    mesh = new Mesh("sixten", node);
    Camera::instance().positionIs(Vector3(11.1429, -5.2408, 10.2673));
    Camera::instance().rotationIs(492.8, 718.4);
    shader = new ShaderData("../shaders/phong");

    shader->enableMatrix(MODELVIEW);
    shader->enableMatrix(PROJECTION);
    shader->enableMatrix(NORMAL);

    std::string tex("../textures/armadillo_n.jpg");
    std::string texName("normalMap");
    shader->addTexture(texName, tex);

    mesh->shaderDataIs(shader);
    ASSIMP2MESH::read("../models/armadillo.3ds", "0", mesh);
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

    initParticleSystem();
    initGL();

    glutMotionFunc(mouseFunc);
    glutPassiveMotionFunc(mouseMoveFunc);

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
    glutMainLoop();
    return 0;
}


