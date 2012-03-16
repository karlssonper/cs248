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
float lastTime = 0;

static void Reshape(int w, int h)
{
    Graphics::instance().viewportIs(w,h);
    Camera::instance().aspectRatioIs(static_cast<float>(w)/h);
}

static void KeyPressed(unsigned char key, int x, int y) {
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
        case 'b':
            Camera::instance().shake(2.f, 4.f);
            break;

    }
}

static void KeyReleased(unsigned char key, int x, int y) {
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

static void MouseFunc(int x,int y)
{
    int dx = x - Engine::instance().mouseX();
    int dy = y - Engine::instance().mouseY();
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
    Camera::instance().yaw(1.6*dx);
    Camera::instance().pitch(1.6*dy);
}

static void MouseMoveFunc(int x,int y)
{
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
}

static void GameLoop()
{
    //the heart
    lastTime = currentTime;
    currentTime = (float)glutGet(GLUT_ELAPSED_TIME) / 1000.f;
    float frameTime = currentTime - lastTime;
    std::cout << std::endl;
    //std::cout << "lastTime: " << lastTime << std::endl;
    //std::cout << "currentTime: " << currentTime << std::endl;
    //std::cout << "frameTime: " << frameTime << std::endl;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Camera::instance().BuildViewMatrix();
    mesh->display();
    CUDA::Ocean::performIFFT(currentTime, false);
    CUDA::Ocean::updateVBO(false);
    CUDA::Ocean::display();

    Camera::instance().updateShake(frameTime);

    //std::cout << "Yaw: " << Camera::instance().yaw() << std::endl;
    //std::cout << "Pitch: " << Camera::instance().pitch() << std::endl;

    glutSwapBuffers();

    //1. Update the global time

    //2. Update camera and view matrix

    //3. Get input (shoot weapon etc)

    //4. Move objects, update particelsystem (CUDA), update ocean(CUDA)

    //5. Update all nodes (update their global model matrix)

    //6. Collision test, trigger stuff

    //7. Render shadow map

    //8. Render regular scene (with shadowmap)

    //9. Render Velocity buffer

    //10. Render CoC?

    //11. Let CUDA blur intensities in regular 1st pass to create bloom map

    //12. Render 2nd pass, combine


}

Engine::Engine()
{
    state_ = NOT_INITIATED;
}

void Engine::init(int argc, char **argv, const char * _titlee, int _width, int _height)
{
    //int argc = 1;
    //char **argv;
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize(_width, _height);
    glutCreateWindow("tests/engine");
    glutReshapeFunc(Reshape);
    glutKeyboardFunc(KeyPressed);
    glutKeyboardUpFunc(KeyReleased);
    glutMotionFunc(MouseFunc);
    glutPassiveMotionFunc(MouseMoveFunc);
    glutDisplayFunc(GameLoop);
    glutIdleFunc(GameLoop);

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


    Graphics::instance().createTextureToFBO("shadow", shadowTex_,
            shadowFB_, 1028, 1028);

    std::vector<unsigned int> colorTex(4);
    std::vector<std::string> colorTexNames;
    colorTexNames.push_back("Phong");
    colorTexNames.push_back("Bloom");
    colorTexNames.push_back("Motion");
    colorTexNames.push_back("CoC");

    Graphics::instance().createTextureToFBO(colorTexNames, colorTex,
            firstPassFB_, firstPassDepthFB_, 1028, 1028);

    Camera::instance().maxYawIs(492.8+45.0);
    Camera::instance().minYawIs(492.8-45.0);
    Camera::instance().maxPitchIs(718.4+10.0);
    Camera::instance().minPitchIs(718.4-10.0);


}

void Engine::BuildQuad()
{
    std::vector<QuadVertex> quadVertices(4);
    quadVertices[0].pos[0] = 0.0f;
    quadVertices[0].pos[1] = 0.0f;
    quadVertices[0].pos[2] = 0.0f;
    quadVertices[0].texCoords[0] = 0.0f;
    quadVertices[0].texCoords[1] = 0.0f;
    quadVertices[1].pos[0] = 1.0f;
    quadVertices[1].pos[1] = 0.0f;
    quadVertices[1].pos[2] = 0.0f;
    quadVertices[1].texCoords[0] = 1.0f;
    quadVertices[1].texCoords[1] = 0.0f;
    quadVertices[2].pos[0] = 1.0f;
    quadVertices[2].pos[1] = 1.0f;
    quadVertices[2].pos[2] = 0.0f;
    quadVertices[2].texCoords[0] = 1.0f;
    quadVertices[2].texCoords[1] = 1.0f;
    quadVertices[3].pos[0] = 0.0f;
    quadVertices[3].pos[1] = 1.0f;
    quadVertices[3].pos[2] = 0.0f;
    quadVertices[3].texCoords[0] = 0.0f;
    quadVertices[3].texCoords[1] = 1.0f;

    std::vector<unsigned int> quadIdx(6);
    quadIdx[0] = 0;
    quadIdx[1] = 1;
    quadIdx[2] = 2;
    quadIdx[3] = 0;
    quadIdx[4] = 2;
    quadIdx[5] = 3;

    std::string quadName("quad");
    Graphics & g = Graphics::instance();
    g.buffersNew(quadName, quadVAO_, quadVBO_, quadIdxVBO_);
    g.geometryIs(quadVBO_,quadIdxVBO_, quadVertices,quadIdx,VBO_STATIC);

    const int stride = sizeof(QuadVertex);

    quadShader_ = g.shader("../shaders/second");

    std::string posStr("positionIn");
    std::string texStr("texcoordIn");

    int posLoc = g.shaderAttribLoc(quadShader_ , posStr);
    int texLoc = g.shaderAttribLoc(quadShader_ , texStr);

    g.bindGeometry(quadShader_, quadVAO_, quadVBO_, 3, stride, posLoc, 0);
    g.bindGeometry(quadShader_, quadVAO_, quadVBO_, 2, stride, texLoc, 12);
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
