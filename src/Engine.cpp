/*
 * Engine.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */
#include <sstream>
#include <time.h>

//Singletons
#include "Engine.h"
#include "Graphics.h"
//#include "Sound.h"

//Elements
#include "Camera.h"
#include "Node.h"
#include "Mesh.h"
#include "ShaderData.h"
#include "Target.h"
#include "HitBox.h"

//CUDA
#include "cuda/Ocean.cuh"

//Wrappers
#include "wrappers/Assimp2Mesh.h"
#include "wrappers/FreeImage2Tex.h"

//Important to include glut AFTER OpenGL
#include <GL/glut.h>



static void Reshape(int w, int h)
{
    Graphics::instance().viewportIs(w,h);
    Engine::instance().camera()->aspectRatioIs(static_cast<float>(w)/h);
    Engine::instance().widthIs(w);
    Engine::instance().heightIs(h);
}

static void KeyPressed(unsigned char key, int x, int y) {
    switch (key){
        case 27:
            // Important to clean up Engine first (delete any loaded meshes)
            // so that we don't try to delete
            // any Meshes that don't exist later.
            Engine::instance().cleanUp();
            Graphics::instance().cleanUp();
            exit(0);
        case 'w':
            Engine::instance().camera()->move(0.5);
            break;
        case 's':
            Engine::instance().camera()->move(-0.5);
            break;
        case 'a':
            Engine::instance().camera()->strafe(-0.5);
            break;
        case 'd':
            Engine::instance().camera()->strafe(0.5);
            break;
        case 'b':
            //Sound::instance().play(Sound::EXPLOSION, Vector3(0,0,0));
            Engine::instance().camera()->shake(2.f, 4.f);
            break;
        case '1':
            Engine::instance().renderTexture(1.0f);
            break;
        case '2':
            Engine::instance().renderTexture(2.0f);
            break;
        case '3':
            Engine::instance().renderTexture(3.0f);
            break;
        case '4':
            Engine::instance().renderTexture(4.0f);
            break;
        case '5':
            Engine::instance().renderTexture(5.0f);
            break;
        case 'q':
            Engine::instance().changeCamera();
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
    Engine::instance().camera()->yaw(1.6*dx);
    Engine::instance().camera()->pitch(1.6*dy);
}

static void MouseMoveFunc(int x,int y)
{
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
}

static void GameLoop()
{
    //the heart
    float currentTime = (float)glutGet(GLUT_ELAPSED_TIME) / 1000.f;

    Engine::instance().renderFrame(currentTime);

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

    //8. Render buffers: Phong, Bloom, Motion, CoC(?)

    //9. Gaussian blur for Bloom map

    //10. Render 2nd pass, combine

}

Engine::Engine()
{
    state_ = NOT_INITIATED;
}

Engine::~Engine() {
    std::cout << "~Engine()" << std::endl;
}

void Engine::init(int argc, char **argv,
                  const char * _titlee,
                  int _width,
                  int _height)
{
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
    currentTime_ = 0;
    nextSpawn_ = 0.f;
    state_ = RUNNING;
    srand(1986);
    root_ = new Node("root");
}

void Engine::start()
{
    glutMainLoop();
}

void Engine::renderTexture(float v)
{
    static std::string debug("debug");
    float* val = quadShader_->floatData(debug);
    *val = v;
}

void Engine::changeCamera()
{
    if (activeCam_ == gameCam_) {
        updateCamView_ = false;
        activeCam_ = lightCam_;
    } else {
        updateCamView_ = true;
        activeCam_ = gameCam_;
    }
    for (ShaderMap::iterator it = shaders_.begin(); it != shaders_.end(); ++it){
        Matrix4 * projection = it->second->stdMatrix4Data(PROJECTION);
        *projection = activeCam_->projectionMtx();
    }
}

void Engine::loadResources(const char * _file)
{
    //same as cudaoceantest

    xzBoundsIs(0.f, 100.f, 0.f ,100.f);
    nrTargetsIs(5);
    targetSpawnRateIs(3.f);

    //Order here is important.
    LoadCameras();
    LoadLight();
    CreateFramebuffer();
    BuildQuad();
    BuildSkybox();
    LoadTargets();
    LoadOcean();
}

void Engine::cleanUp() {
    //delete node;
    //delete mesh;
    //delete shader;
}

void Engine::renderFrame(float _currentTime)
{
    //todo remove GL call
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float lastTime = currentTime_;
    currentTime_ = _currentTime;
    float frameTime = currentTime_ - lastTime;

    if (updateCamView_) {
        activeCam_->BuildViewMatrix();
    }
    activeCam_->updateShake(frameTime);

    CUDA::Ocean::performIFFT(currentTime_, false);
    CUDA::Ocean::updateVBO(false);
    SpawnTargets();

    root_->update();

    UpdateTargets(frameTime);

    RenderShadowMap();

    RenderFirstPass();

    RenderSecondPass();
}

void Engine::RenderShadowMap()
{
    Graphics::instance().enableFramebuffer(shadowFB_, shadowSize_, shadowSize_);
    for (MeshMap::const_iterator it =meshes_.begin(); it!= meshes_.end();++it) {
        it->second->displayShadowPass(shadowShader_);
    }
    //CUDA::Ocean::dsplay();
    Graphics::instance().disableFramebuffer();
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

    //Skybox
    Matrix4 * modelView = skyBoxShader_->stdMatrix4Data(MODELVIEW);
    *modelView = camera()->viewMtx();

    Graphics::instance().drawIndices(skyboxVAO_,
                                     skyboxIdxVBO_,
                                     36,
                                     skyBoxShader_);

    for (MeshMap::const_iterator it =meshes_.begin(); it!= meshes_.end();++it) {
        it->second->display();
    }
    CUDA::Ocean::display();

    Graphics::instance().disableFramebuffer();
}



void Engine::RenderSecondPass()
{
    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6, quadShader_);
}

void Engine::BuildQuad()
{
    std::vector<QuadVertex> v(4);
    v[0].pos[0] = -1.0f; v[0].pos[1] = -1.0f; v[0].pos[2] = 0.0f;
    v[0].texCoords[0] = 0.0f; v[0].texCoords[1] = 0.0f;
    v[1].pos[0] = 1.0f; v[1].pos[1] = -1.0f; v[1].pos[2] = 0.0f;
    v[1].texCoords[0] = 1.0f; v[1].texCoords[1] = 0.0f;
    v[2].pos[0] = 1.0f; v[2].pos[1] = 1.0f; v[2].pos[2] = 0.0f;
    v[2].texCoords[0] = 1.0f; v[2].texCoords[1] = 1.0f;
    v[3].pos[0] = -1.0f; v[3].pos[1] = 1.0f; v[3].pos[2] = 0.0f;
    v[3].texCoords[0] = 0.0f; v[3].texCoords[1] = 1.0f;

    std::vector<unsigned int> quadIdx(6);
    quadIdx[0] = 0; quadIdx[1] = 1; quadIdx[2] = 2;
    quadIdx[3] = 0; quadIdx[4] = 2; quadIdx[5] = 3;

    std::string quadName("quad");
    Graphics & g = Graphics::instance();
    g.buffersNew(quadName, quadVAO_, quadVBO_, quadIdxVBO_);
    g.geometryIs(quadVBO_,quadIdxVBO_, v,quadIdx,VBO_STATIC);

    const int stride = sizeof(QuadVertex);

    std::string shaderStr("../shaders/second");
    quadShader_ = new ShaderData(shaderStr);
    unsigned int sID = quadShader_->shaderID();

    std::vector<std::string> colorTexNames;
    colorTexNames.push_back("Phong");
    colorTexNames.push_back("Bloom");
    colorTexNames.push_back("Motion");
    colorTexNames.push_back("CoC");
    colorTexNames.push_back("shadow");

    std::vector<std::string> shaderTexNames;
    shaderTexNames.push_back("phongTex");
    shaderTexNames.push_back("bloomTex");
    shaderTexNames.push_back("motionTex");
    shaderTexNames.push_back("cocTex");
    shaderTexNames.push_back("shadowTex");

    quadShader_->addTexture(shaderTexNames[0], colorTexNames[0]);
    quadShader_->addTexture(shaderTexNames[1], colorTexNames[1]);
    quadShader_->addTexture(shaderTexNames[2], colorTexNames[2]);
    quadShader_->addTexture(shaderTexNames[3], colorTexNames[3]);
    quadShader_->addTexture(shaderTexNames[4], colorTexNames[4]);

    quadShader_->addFloat("debug",1.0f);

    std::string posStr("positionIn");
    std::string texStr("texcoordIn");

    int posLoc = g.shaderAttribLoc(sID , posStr);
    int texLoc = g.shaderAttribLoc(sID , texStr);

    g.bindGeometry(sID, quadVAO_, quadVBO_, 3, stride, posLoc, 0);
    g.bindGeometry(sID, quadVAO_, quadVBO_, 2, stride, texLoc, 12);
}

void Engine::BuildSkybox()
{
    std::vector<SkyboxVertex> v(8);
    const float scale = 100;
    v[0].pos[0] = -scale; v[0].pos[1] = -scale; v[0].pos[2] = -scale;
    v[1].pos[0] = scale;  v[1].pos[1] = -scale; v[1].pos[2] = -scale;
    v[2].pos[0] = -scale; v[2].pos[1] = -scale; v[2].pos[2] = scale;
    v[3].pos[0] = scale;  v[3].pos[1] = -scale; v[3].pos[2] = scale;
    v[4].pos[0] = -scale; v[4].pos[1] = scale;  v[4].pos[2] = -scale;
    v[5].pos[0] = scale;  v[5].pos[1] = scale;  v[5].pos[2] = -scale;
    v[6].pos[0] = -scale; v[6].pos[1] = scale;  v[6].pos[2] = scale;
    v[7].pos[0] = scale;  v[7].pos[1] = scale;  v[7].pos[2] = scale;

    std::vector<unsigned int> i(3*6*2);
    i[0]  = 0; i[1]  = 1; i[2]  = 2;
    i[3]  = 1; i[4]  = 2; i[5]  = 3;
    i[6]  = 4; i[7]  = 5; i[8]  = 6;
    i[9]  = 5; i[10] = 6; i[11] = 7;
    i[12] = 1; i[13] = 3; i[14] = 7;
    i[15] = 1; i[16] = 5; i[17] = 7;
    i[18] = 0; i[19] = 2; i[20] = 6;
    i[21] = 0; i[22] = 4; i[23] = 6;
    i[24] = 2; i[25] = 3; i[26] = 7;
    i[27] = 2; i[28] = 6; i[29] = 7;
    i[30] = 0; i[31] = 1; i[32] = 5;
    i[33] = 0; i[34] = 4; i[35] = 5;

    std::string boxName("skybox");
    Graphics & g = Graphics::instance();
    g.buffersNew(boxName, skyboxVAO_, skyboxVBO_, skyboxIdxVBO_);
    g.geometryIs(skyboxVBO_,skyboxIdxVBO_, v,i,VBO_STATIC);

    std::string skyboxShaderStr("../shaders/skybox");
    skyBoxShader_ = new ShaderData(skyboxShaderStr);
    skyBoxShader_->enableMatrix(PROJECTION);
    Matrix4 * projection = skyBoxShader_->stdMatrix4Data(PROJECTION);
    *projection = camera()->projectionMtx();
    skyBoxShader_->enableMatrix(MODELVIEW);
    std::string cubeMapStr("CubeMap");
    std::string cubeMapShaderStr("skyboxTex");
    std::vector<std::string> cubeMapTexs(6);
    cubeMapTexs[0] = std::string("../textures/POSITIVE_X.png");
    cubeMapTexs[1] = std::string("../textures/NEGATIVE_X.png");
    cubeMapTexs[2] = std::string("../textures/POSITIVE_Y.png");
    cubeMapTexs[3] = std::string("../textures/NEGATIVE_Y.png");
    cubeMapTexs[4] = std::string("../textures/POSITIVE_Z.png");
    cubeMapTexs[5] = std::string("../textures/NEGATIVE_Z.png");
   // skyBoxShader_->addCubeTexture(cubeMapShaderStr, cubeMapStr, cubeMapTexs);

    const int stride = sizeof(SkyboxVertex);
    unsigned int sID = skyBoxShader_->shaderID();
    std::string posStr("positionIn");
    int posLoc = g.shaderAttribLoc(sID , posStr);
    g.bindGeometry(sID, skyboxVAO_, skyboxVBO_, 3, stride, posLoc, 0);
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

void Engine::xzBoundsIs(float _xMin, float _xMax, float _zMin, float _zMax) {
    xMin_ = _xMin;
    xMax_ = _xMax;
    zMin_ = _zMin;
    zMax_ = _zMax;
}

void Engine::CreateFramebuffer()
{
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
}

void Engine::LoadCameras()
{
    gameCam_ = new Camera();
    gameCam_->projectionIs(45.f, 1.f, 1.f, 10000.f);
    gameCam_->positionIs(Vector3(11.1429, -5.2408, 10.2673));
    gameCam_->rotationIs(492.8, 718.4);
    /*gameCam_->maxYawIs(492.8+45.0);
    gameCam_->minYawIs(492.8-45.0);
    gameCam_->maxPitchIs(718.4+10.0);
    gameCam_->minPitchIs(718.4-10.0);*/
    activeCam_ = gameCam_;
    updateCamView_ = true;

    freeCam_ = new Camera();
    freeCam_->projectionIs(45.f, 1.f, 1.f, 10000.f);
    freeCam_->positionIs(Vector3(11.1429, -5.2408, 10.2673));
    freeCam_->rotationIs(492.8, 718.4);

    lightCam_ = new Camera();
}

void Engine::LoadLight()
{
    shadowSize_ = 1024;
    Graphics::instance().createTextureToFBO("shadow", shadowTex_,
            shadowFB_, shadowSize_, shadowSize_);
    std::string shadowShaderStr("../shaders/shadow");
    shadowShader_ = new ShaderData(shadowShaderStr);
    shadowShader_->enableMatrix(PROJECTION);
    shadowShader_->enableMatrix(MODELVIEW);
    lightCam_->lookAt(
            Vector3(51.0,0.5, 51.0),
            Vector3(50,0,50.0),
            Vector3(0,1.0,0));
    lightCam_->BuildOrthoProjection(
            Vector3(-100,-50,-100),
            Vector3(100,50,100));
    Matrix4 * shadowProj = shadowShader_->stdMatrix4Data(PROJECTION);
    *shadowProj = Engine::instance().lightCamera()->projectionMtx();
}

void Engine::LoadOcean()
{
    CUDA::Ocean::init();
    std::string oceanShaderStr("ocean");
    shaders_[oceanShaderStr] = CUDA::Ocean::oceanShaderData();
    CUDA::Ocean::oceanShaderData()->addTexture("shadowMap", "shadow");
    CUDA::Ocean::oceanShaderData()->addFloat("shadowMapDx", 1.0f / shadowSize_);
}

void Engine::LoadTargets() {

    std::string phongStr("phong");
    ShaderData * shader = new ShaderData("../shaders/phong");
    shaders_[phongStr] = shader;
    shader->enableMatrix(MODELVIEW);
    shader->enableMatrix(NORMAL);
    shader->enableMatrix(PROJECTION);
    Matrix4 * proj = shader->stdMatrix4Data(PROJECTION);
    *proj = Engine::instance().camera()->projectionMtx();

    std::string tex("../textures/Galleon2.jpg");
    std::string texName("diffuseMap");
    shader->addTexture(texName, tex);

    for (unsigned int i = 0; i < nrTargets_; i++) {
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << i ;

        std::string nodeStr("battleCruiserNode"+oss.str());
        Node * node = new Node(nodeStr);
        nodes_[nodeStr] = node;
        node->parentIs(root_);
        //float startX = Random::randomFloat(xMin_, xMax_);
        //node->translate(Vector3(startX, 0.f, zMax_));
        //node->rotateY(180.0f);

        std::string meshStr("battleCruiser"+oss.str());
        Mesh * mesh = new Mesh(meshStr, node);
        meshes_[meshStr] = mesh;
        mesh->node()->rotateY(180.0f);

        mesh->showIs(false);

        mesh->shaderDataIs(shader);
        ASSIMP2MESH::read("../models/Galleon.3ds", "galleon", mesh, 0.3f);

        std::string targetStr("battleCruiserTarget"+oss.str());
        Target * target = new Target(targetStr, mesh, 100.f);
        target->speedIs(Vector3(0.0f, 0.f, 15.0f));
        target->hitBox()->p0.print();
        target->hitBox()->p1.print();
        targets_.push_back(target);
    }
}

void Engine::nrTargetsIs(unsigned int _nrTargets) {
    nrTargets_ = _nrTargets;
}

void Engine::targetSpawnRateIs(float _targetSpawnRate) {
    targetSpawnRate_ = _targetSpawnRate;
}

void Engine::UpdateTargets(float _frameTime) {

    nextSpawn_ -= _frameTime;

    std::vector<Target*>::iterator it;
    for (it=targets_.begin(); it!=targets_.end(); it++) {

        if ( (*it)->active() ) {
            (*it)->updatePos(_frameTime);
            (*it)->updateHitBox();
            if ( (*it)->midPoint().z < zMin_ ) {
             
                (*it)->activeIs(false);
                (*it)->mesh()->showIs(false);
      
            }

        }

    }
}

void Engine::SpawnTargets() {

    if (nextSpawn_ < 0.f) {
        std::vector<Target*>::iterator it;
        for (it=targets_.begin(); it!=targets_.end(); it++) {
            
            if ( !(*it)->active() ) {
                //std::cout << "Activating " << (*it)->name() << std::endl;

                float startX = Random::randomFloat(xMin_, xMax_);
                Vector3 startPos(startX, 0.f, zMax_);
                Vector3 currentPos = (Vector3((*it)->midPoint().x, 
                                              0.f,
                                              (*it)->midPoint().z));
                (*it)->mesh()->node()->translate(currentPos-startPos);
                (*it)->activeIs(true);
                (*it)->mesh()->showIs(true);

                break;
            }
        }

        nextSpawn_ = targetSpawnRate_;
    }

}



