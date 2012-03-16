/*
 * Engine.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */
//Singletons
#include "Engine.h"
#include "Camera.h"
#include "Graphics.h"

//Elements
#include "Node.h"
#include "Mesh.h"

//CUDA
#include "cuda/Ocean.cuh"

//Wrappers
#include "wrappers/Assimp2Mesh.h"
#include "wrappers/FreeImage2Tex.h"

//Important to include glut AFTER OpenGL
#include <GL/glut.h>

//remove
Mesh * mesh;
ShaderData * shader;
Node * node;
float currentTime = 0;

static void reshape(int w, int h)
{
    Graphics::instance().viewportIs(w,h);
    Camera::instance().aspectRatioIs(static_cast<float>(w)/h);
}

static void keyPressed(unsigned char key, int x, int y) {
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
    }
}

static void keyReleased(unsigned char key, int x, int y) {
    switch (key){
        case 27:
            exit(0);
        case 'w':
            //Camera::instance().move(0.5);
            break;
        case 's':
            //Camera::instance().move(-0.5);
            break;
        case 'a':
            //Camera::instance().strafe(-0.5);
            break;
        case 'd':
            //Camera::instance().strafe(0.5);
            break;
    }
}

static void mouseFunc(int x,int y)
{
    int dx = x - Engine::instance().mouseX();
    int dy = y - Engine::instance().mouseY();
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
    Camera::instance().yaw(1.6*dx);
    Camera::instance().pitch(1.6*dy);
}

static void mouseMoveFunc(int x,int y)
{
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
}

static void gameLoop()
{
    //the heart
    currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.f;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera::instance().BuildViewMatrix();
    mesh->display();
    CUDA::Ocean::performIFFT(currentTime, false);
    CUDA::Ocean::updateVBO(false);
    CUDA::Ocean::display();
    glutSwapBuffers();
}

Engine::Engine()
{
    state_ = NOT_INITIATED;
}

void Engine::init(const char * _titlee, int _width, int _height)
{
    int argc = 1;
    char **argv;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(_width, _height);
    glutCreateWindow("tests/engine");
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyPressed);
    glutKeyboardUpFunc(keyReleased);
    glutMotionFunc(mouseFunc);
    glutPassiveMotionFunc(mouseMoveFunc);
    glutDisplayFunc(gameLoop);
    glutIdleFunc(gameLoop);

    state_ = RUNNING;
}

void Engine::loadResources(const char * _file)
{
    //same as cudaoceantest

    node = new Node("sixtenNode");
    mesh = new Mesh("sixten", node);
    Camera::instance().projectionIs(45.f, 1.f, 1.f, 100.f);
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
    CUDA::Ocean::init();
}

void Engine::mouseXIs(int x)
{
    mouseX_ = x;
}

void Engine::mouseYIs(int y)
{
    mouseY_ = y;
}

void Engine::start()
{
    glutMainLoop();
}
