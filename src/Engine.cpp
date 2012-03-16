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
#include "ShaderData.h"

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

static void Reshape(int w, int h)
{
    Graphics::instance().viewportIs(w,h);
    Camera::instance().aspectRatioIs(static_cast<float>(w)/h);
    Engine::instance().widthIs(w);
    Engine::instance().heightIs(h);
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
    currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.f;
    Camera::instance().BuildViewMatrix();
    Engine::instance().renderFrame();
    glutSwapBuffers();

    //1. Update the global time

    //2. Update camera and view matrix

    //3. Get input (shoot weapon etc)

    //4. Move objects, update particelsystem (CUDA), update ocean(CUDA)

    //5. Update all nodes (update their global model matrix)

    //6. Collision test, trigger stuff

    //7. Render shadow map

    //8. Render buffers: Phong, Bloom, Motion, CoC(?)

    //9. Gaussian blur for Bloom map

    //10. Render 2nd pass, combine
}

Engine::Engine()
{
    state_ = NOT_INITIATED;
}

void Engine::init(const char * _titlee, int _width, int _height)
{
    int argc = 1;
    char **argv;
    widthIs(_width);
    heightIs(_height);
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

    phongTex_ = colorTex[0];
    bloomTex_ = colorTex[1];
    motionTex_ = colorTex[2];
    cocTex_ = colorTex[3];

    Graphics::instance().createTextureToFBO(colorTexNames, colorTex,
            firstPassFB_, firstPassDepthFB_, width(), height());

    BuildQuad();

}

void Engine::BuildQuad()
{
    std::vector<QuadVertex> quadVertices(4);
    quadVertices[0].pos[0] = -1.0f;
    quadVertices[0].pos[1] = -1.0f;
    quadVertices[0].pos[2] = 0.0f;
    quadVertices[0].texCoords[0] = 0.0f;
    quadVertices[0].texCoords[1] = 0.0f;
    quadVertices[1].pos[0] = 1.0f;
    quadVertices[1].pos[1] = -1.0f;
    quadVertices[1].pos[2] = 0.0f;
    quadVertices[1].texCoords[0] = 1.0f;
    quadVertices[1].texCoords[1] = 0.0f;
    quadVertices[2].pos[0] = 1.0f;
    quadVertices[2].pos[1] = 1.0f;
    quadVertices[2].pos[2] = 0.0f;
    quadVertices[2].texCoords[0] = 1.0f;
    quadVertices[2].texCoords[1] = 1.0f;
    quadVertices[3].pos[0] = -1.0f;
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

    std::string shaderStr("../shaders/second");
    quadShader_ = new ShaderData(shaderStr);
    unsigned int sID = quadShader_->shaderID();

    std::vector<std::string> colorTexNames;
    colorTexNames.push_back("Phong");
    colorTexNames.push_back("Bloom");
    colorTexNames.push_back("Motion");
    colorTexNames.push_back("CoC");

    std::vector<std::string> shaderTexNames;
    shaderTexNames.push_back("phongTex");
    shaderTexNames.push_back("bloomTex");
    shaderTexNames.push_back("motionTex");
    shaderTexNames.push_back("cocTex");

    quadShader_->addTexture(shaderTexNames[0], colorTexNames[0]);
    quadShader_->addTexture(shaderTexNames[1], colorTexNames[1]);
    quadShader_->addTexture(shaderTexNames[2], colorTexNames[2]);
    quadShader_->addTexture(shaderTexNames[3], colorTexNames[3]);

    std::string posStr("positionIn");
    std::string texStr("texcoordIn");

    int posLoc = g.shaderAttribLoc(sID , posStr);
    int texLoc = g.shaderAttribLoc(sID , texStr);

    g.bindGeometry(sID, quadVAO_, quadVBO_, 3, stride, posLoc, 0);
    g.bindGeometry(sID, quadVAO_, quadVBO_, 2, stride, texLoc, 12);
}

void Engine::renderFrame()
{
    //todo remove GL call
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    RenderShadowMap();

    RenderFirstPass();

    RenderSecondPass();
}

void Engine::RenderShadowMap()
{

}

void Engine::RenderFirstPass()
{
    Graphics::instance().enableFramebuffer(
                                            firstPassDepthFB_,
                                            firstPassFB_,
                                            0,
                                            4,
                                            width(),
                                            height());
    mesh->display();
    CUDA::Ocean::performIFFT(currentTime, false);
    CUDA::Ocean::updateVBO(false);
    CUDA::Ocean::display();

    Graphics::instance().disableFramebuffer();
}

void Engine::RenderSecondPass()
{
    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6, quadShader_);
}

void Engine::mouseXIs(int x)
{
    mouseX_ = x;
}

void Engine::mouseYIs(int y)
{
    mouseY_ = y;
}

void Engine::widthIs(int _width)
{
    width_ = _width;
}

void Engine::heightIs(int _height)
{
    height_ = _height;
}

void Engine::start()
{
    glutMainLoop();
}
