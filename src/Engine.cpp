/*
* Engine.cpp
*
* Created on: Mar 2, 2012
* Author: per
*/
#include <sstream>
#include <time.h>

//Singletons
#include "Engine.h"
#include "Graphics.h"
#include "Sound.h"

//Elements
#include "Camera.h"
#include "Node.h"
#include "Mesh.h"
#include "ShaderData.h"
#include "Target.h"
#include "MeshedWeapon.h"
#include "MeshedProjectile.h"
#include "HitBox.h"
#include "ParticleSystem.h"
#include "cuda/Emitter.cuh"

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

    Vector3 direction;
    float pitch, yaw;

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
            direction = Engine::instance().camera()->viewVector();
            pitch = -Engine::instance().camera()->pitch();
            yaw = -Engine::instance().camera()->yaw()+90.f;
            Engine::instance().rocketLauncher()->fire(direction, pitch, yaw);

            //Engine::instance().camera()->position().print();
            //std::cout << "Camera pitch " << Engine::instance().camera()->pitch() << std::endl;
            //std::cout << "Camera yaw " << Engine::instance().camera()->yaw() << std::endl;
            //Engine::instance().rocketLauncher()->projectiles().at(0)->rotationNode()->rotateX(-Engine::instance().camera()->pitch());
            //Engine::instance().rocketLauncher()->projectiles().at(0)->rotationNode()->rotateY(-Engine::instance().camera()->yaw()+90.f);
            break;
        //ZIMMERMAN!!!
        case 'z':
            Sound::instance().play(Sound::THEME, Vector3(0,0,0));
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

    Vector3 direction;
    float pitch, yaw;
    direction = Engine::instance().camera()->viewVector();
    pitch = -Engine::instance().camera()->pitch();
    yaw = -Engine::instance().camera()->yaw()+90.f;
    Engine::instance().rocketLauncher()->fire(direction, pitch, yaw);
}

static void MouseMoveFunc(int x,int y)
{

    int dx = x - Engine::instance().mouseX();
    int dy = y - Engine::instance().mouseY();
    Engine::instance().mouseXIs(x);
    Engine::instance().mouseYIs(y);
    Engine::instance().camera()->yaw(0.3*dx);
    Engine::instance().camera()->pitch(0.3*dy);

    //Engine::instance().mouseXIs(x);
    //Engine::instance().mouseYIs(y);
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
    glutSetCursor(GLUT_CURSOR_NONE);
    currentTime_ = 0;
    nextSpawn_ = 0.f;
    state_ = RUNNING;
    Sound::instance().listenerPositionIs(Vector3(0.f, 0.f, 0.f));
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
    LoadOcean();
    CreateFramebuffer();
    BuildQuad();
    BuildSkybox();
    LoadTargets();
    loadWeapons();
    initParticleSystems();
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


    Matrix4 * prevViewProj = quadShader_->stdMatrix4Data(PREVVIEWPROJECTION);
    *prevViewProj = activeCam_->projectionMtx() * activeCam_->viewMtx();

    if (updateCamView_) {
        activeCam_->BuildViewMatrix();
    }

    UpdateDOF();

    Matrix4 * invViewProj = quadShader_->stdMatrix4Data(INVERSEVIEWPROJECTION);
    *invViewProj =
            (activeCam_->projectionMtx() * activeCam_->viewMtx()).inverse();

    activeCam_->updateShake(frameTime);

    CUDA::Ocean::performIFFT(currentTime_, false);
    CUDA::Ocean::updateVBO(false);

    SpawnTargets();
    root_->update();
    UpdateTargets(frameTime);

    rocketLauncher_->positionIs(Engine::instance().camera()->worldPos(2.f));
    updateProjectiles(frameTime);

    updateParticles(frameTime);
    
    RenderShadowMap();
    RenderFirstPass();
    BlurTextures();
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
                                            firstPassFB_,
                                            3,
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

    Graphics::instance().enableFramebuffer(
                                            softParticlesDepthFB_,
                                            softParticlesFB_,
                                            0,
                                            1,
                                            width(),
                                            height());
    displayParticles();

    Graphics::instance().disableFramebuffer();
}



void Engine::RenderSecondPass()
{
    Graphics::instance().enableFramebuffer(
                                            secondPassDepthFB_,
                                            secondPassFB_,
                                            0,
                                            2,
                                            width(),
                                            height()
                                           );
    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6, quadShader_);

    Graphics::instance().enableFramebuffer(
                                            horBlurDepthFB_,
                                            horBlurFB_,
                                            0,
                                            2,
                                            width(),
                                            height()
                                           );
    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6,
                                     horDOFShader_);
    Graphics::instance().disableFramebuffer();

    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6,
                                     vertDOFShader_);
}

void Engine::BlurTextures()
{
    Graphics::instance().enableFramebuffer(
                                            horBlurDepthFB_,
                                            horBlurFB_,
                                            0,
                                            2,
                                            width(),
                                            height()
                                           );
    Graphics::instance().drawIndices(quadVAO_, quadIdxVBO_, 6,
                                     horizontalGaussianShader_);
    Graphics::instance().disableFramebuffer();
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
    colorTexNames.push_back("Particles");
    colorTexNames.push_back("Bloom2");
    colorTexNames.push_back("CoC2");
    colorTexNames.push_back("shadow");
    colorTexNames.push_back("../textures/hud.png");
    colorTexNames.push_back("depth");


    std::vector<std::string> shaderTexNames;
    shaderTexNames.push_back("phongTex");
    shaderTexNames.push_back("particlesTex");
    shaderTexNames.push_back("bloomTex");
    shaderTexNames.push_back("cocTex");
    shaderTexNames.push_back("shadowTex");
    shaderTexNames.push_back("hudTex");
    shaderTexNames.push_back("depthTex");


    quadShader_->addTexture(shaderTexNames[0], colorTexNames[0]);
    quadShader_->addTexture(shaderTexNames[1], colorTexNames[1]);
    quadShader_->addTexture(shaderTexNames[2], colorTexNames[2]);
    quadShader_->addTexture(shaderTexNames[3], colorTexNames[3]);
    quadShader_->addTexture(shaderTexNames[4], colorTexNames[4]);
    quadShader_->addTexture(shaderTexNames[5], colorTexNames[5]);
    quadShader_->addTexture(shaderTexNames[6], colorTexNames[6]);

    quadShader_->enableMatrix(INVERSEVIEWPROJECTION);
    quadShader_->enableMatrix(PREVVIEWPROJECTION);

    quadShader_->addFloat("debug",1.0f);
    quadShader_->addFloat("texDx", 1.0f / height());

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
    const float scale = 300;
    v[0].pos[0] = 0.5*-scale; v[0].pos[1] = -scale; v[0].pos[2] = -scale;
    v[1].pos[0] = 0.5*scale; v[1].pos[1] = -scale; v[1].pos[2] = -scale;
    v[2].pos[0] = 0.5*-scale; v[2].pos[1] = -scale; v[2].pos[2] = scale;
    v[3].pos[0] = 0.5*scale; v[3].pos[1] = -scale; v[3].pos[2] = scale;
    v[4].pos[0] = 0.5*-scale; v[4].pos[1] = scale; v[4].pos[2] = -scale;
    v[5].pos[0] = 0.5*scale; v[5].pos[1] = scale; v[5].pos[2] = -scale;
    v[6].pos[0] = 0.5*-scale; v[6].pos[1] = scale; v[6].pos[2] = scale;
    v[7].pos[0] = 0.5*scale; v[7].pos[1] = scale; v[7].pos[2] = scale;

    std::vector<unsigned int> i(3*6*2);
    i[0] = 0; i[1] = 1; i[2] = 2;
    i[3] = 1; i[4] = 2; i[5] = 3;
    i[6] = 4; i[7] = 5; i[8] = 6;
    i[9] = 5; i[10] = 6; i[11] = 7;
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

    skyBoxShader_->addFloat("focalPlane", focalPlane_);
    skyBoxShader_->addFloat("nearBlurPlane", nearBlurPlane_);
    skyBoxShader_->addFloat("farBlurPlane", farBlurPlane_);
    skyBoxShader_->addFloat("maxBlur", maxBlur_);

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
    std::vector<unsigned int> colorTex(3);
    std::vector<std::string> colorTexNames;
    colorTexNames.push_back("Phong");
    colorTexNames.push_back("Bloom");
    colorTexNames.push_back("CoC");

    Graphics::instance().createTextureToFBOTest(colorTexNames, colorTex,
            depthTex_, firstPassFB_, width(), height());

    phongTex_ = colorTex[0];
    bloomTex_ = colorTex[1];
    cocTex_ = colorTex[2];

    std::vector<unsigned int> particlesTex(1);
    std::vector<std::string> particleTexNames;
    particleTexNames.push_back("Particles");
    Graphics::instance().createTextureToFBO(particleTexNames, particlesTex,
            softParticlesFB_, softParticlesDepthFB_, width(), height());

    std::vector<unsigned int> horBlurTex(2);
    std::vector<std::string> horBlurTexNames;
    horBlurTexNames.push_back("Bloom2");
    horBlurTexNames.push_back("CoC2");
    Graphics::instance().createTextureToFBO(horBlurTexNames, horBlurTex,
            horBlurFB_, horBlurDepthFB_, width(), height());
    bloom2Tex_ = horBlurTex[0];
    coc2Tex_ = horBlurTex[1];

    horizontalGaussianShader_ = new ShaderData("../shaders/horizontalGauss");
    horizontalGaussianShader_->addTexture("bloomTex", "Bloom");
    horizontalGaussianShader_->addTexture("cocTex", "CoC");
    horizontalGaussianShader_->addFloat("texDx", 1.0f / width());

    std::vector<unsigned int> secondPassTex(2);
    std::vector<std::string> secondPassTexNames;
    secondPassTexNames.push_back("Phong2");
    secondPassTexNames.push_back("CoC3");
    Graphics::instance().createTextureToFBO(secondPassTexNames, secondPassTex,
            secondPassFB_, secondPassDepthFB_, width(), height());

    horDOFShader_ = new ShaderData("../shaders/horDOF");
    horDOFShader_->addTexture("phongTex", "Phong2");
    horDOFShader_->addTexture("cocTex", "CoC3");
    horDOFShader_->addFloat("texDx", 1.0f / width());
    horDOFShader_->addFloat("DOF", 10.0f);

    vertDOFShader_ = new ShaderData("../shaders/vertDOF");
    vertDOFShader_->addTexture("phongTex", "Bloom2");
    vertDOFShader_->addTexture("cocTex", "CoC3");
    vertDOFShader_->addFloat("texDx", 1.0f / width());
    vertDOFShader_->addFloat("DOF", 10.0f);
}

void Engine::LoadCameras()
{
    gameCam_ = new Camera();
    gameCam_->projectionIs(45.f, 1.f, 1.f, 10000.f);
    gameCam_->positionIs(Vector3(25.f, -20.f, 5.f));
    gameCam_->rotationIs(125.f, 15.f);
    gameCam_->maxYawIs(125.f+50.0);
    gameCam_->minYawIs(125.f-50.0);
    gameCam_->maxPitchIs(15.f+15.0);
    gameCam_->minPitchIs(15.f-15.0);
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
    CUDA::Ocean::oceanShaderData()->addTexture("sunReflection",
                                               "../textures/sunReflection.png");
    CUDA::Ocean::oceanShaderData()->addTexture("foamTex",
                                                   "../textures/foam.jpg");
    CUDA::Ocean::oceanShaderData()->addFloat("shadowMapDx", 1.0f / shadowSize_);
    CUDA::Ocean::oceanShaderData()->addFloat("focalPlane", focalPlane_);
    CUDA::Ocean::oceanShaderData()->addFloat("nearBlurPlane", nearBlurPlane_);
    CUDA::Ocean::oceanShaderData()->addFloat("farBlurPlane", farBlurPlane_);
    CUDA::Ocean::oceanShaderData()->addFloat("maxBlur", maxBlur_);

}

void Engine::loadWeapons() {

    // load rocket launcher projectile
    std::string phongStr("phong");
    ShaderData * shader = new ShaderData("../shaders/phong");
    shaders_[phongStr] = shader;
    shader->enableMatrix(MODELVIEW);
    shader->enableMatrix(NORMAL);
    shader->enableMatrix(PROJECTION);
    Matrix4 * proj = shader->stdMatrix4Data(PROJECTION);
    *proj = Engine::instance().camera()->projectionMtx();

    std::string tex("../textures/missile.jpg");
    std::string texName("diffuseMap");
    shader->addTexture(texName, tex);

    std::string nodeStr;

    nodeStr = "rocketTransNode";
    Node * translationNode = new Node(nodeStr);
    nodes_[nodeStr] = translationNode;
    translationNode->parentIs(root_);

    nodeStr = "rocketRotNode";
    Node * rotationNode = new Node(nodeStr);
    nodes_[nodeStr] = rotationNode;
    rotationNode->parentIs(translationNode);

    std::string meshStr("rocketMesh");
    Mesh * mesh = new Mesh(meshStr, rotationNode);
    meshes_[meshStr] = mesh;
    translationNode->rotateY(180.f);

    mesh->showIs(false);

    mesh->shaderDataIs(shader);
    ASSIMP2MESH::read("../models/missile.3ds", "rocket", mesh, 0.4f);

    rocketLauncher_ = new MeshedWeapon( Engine::instance().camera()->worldPos(5),
                                        100.f,
                                        50.f);

    MeshedProjectile * rocket = new MeshedProjectile( Vector3(0.f, 0.f, 0.f),
                                                      Vector3(0.f, 0.f, 0.f),
                                                      rocketLauncher_->power(),
                                                      mesh,
                                                      150.f,
                                                      translationNode,
                                                      rotationNode);
    rocketLauncher_->addProjectile(rocket);
}

void Engine::updateProjectiles(float _dt) {
    std::vector<MeshedProjectile*> projectiles = rocketLauncher_->projectiles();
    for (unsigned int i=0; i<projectiles.size(); ++i) {

        projectiles.at(i)->update(_dt);

        if (projectiles.at(i)->active()) {

            for (unsigned int j=0; j<targets_.size(); ++j) {

                if (targets_.at(j)->active()) {
                    if (projectiles.at(i)->checkCollision(targets_.at(j)->hitBox())) {

                        projectiles.at(i)->resetRotation();
                        targets_.at(j)->explode();
                        projectiles.at(i)->activeIs(false);
                        // TODO: handle power/energy
                    }
                }
            }

        }

        if (projectiles.at(i)->active()) {
            projectiles.at(i)->mesh()->showIs(true);
        } else {
            projectiles.at(i)->mesh()->showIs(false);
        }

    }
}

void Engine::LoadTargets() {

    std::string phongStr("phongTarget");
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
    shader->addFloat("focalPlane", focalPlane_);
    shader->addFloat("nearBlurPlane", nearBlurPlane_);
    shader->addFloat("farBlurPlane", farBlurPlane_);
    shader->addFloat("maxBlur", maxBlur_);

    for (unsigned int i = 0; i < nrTargets_; i++) {
        std::ostringstream oss;
        oss << std::setw(3) << std::setfill('0') << i ;

        std::string nodeStr("battleCruiserNode"+oss.str());
        Node * node = new Node(nodeStr);
        nodes_[nodeStr] = node;
        node->parentIs(root_);

        std::string meshStr("battleCruiser"+oss.str());
        Mesh * mesh = new Mesh(meshStr, node);
        meshes_[meshStr] = mesh;
        mesh->node()->rotateY(180.0f);

        mesh->showIs(false);

        mesh->shaderDataIs(shader);
        ASSIMP2MESH::read("../models/Galleon.3ds", "galleon", mesh, 0.3f);

        std::string targetStr("battleCruiserTarget"+oss.str());
        Target * target = new Target(targetStr, mesh, 100.f);
        target->speedIs(Vector3(0.0f, 0.f, 10.0f));
        target->yOffsetIs(6.0f);
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

    std::vector<std::pair<float,float> > xzPositions;
    for (it=targets_.begin(); it!=targets_.end(); it++) {
        std::pair<float, float> xzPair;
        xzPair.first = (*it)->midPoint().x;
        xzPair.second = (*it)->midPoint().z;
        xzPositions.push_back(xzPair);
    }
    std::vector<float> heights = CUDA::Ocean::height(xzPositions);

    unsigned int i=0;
    for (it=targets_.begin(); it!=targets_.end(); it++) {

        if ( (*it)->active() ) {

            float currentHeight = (*it)->midPoint().y;
            float oceanHeight = heights.at(i);
            (*it)->heightDiffIs(currentHeight - oceanHeight);

            (*it)->updatePos(_frameTime);
            root_->update();
            (*it)->updateHitBox();

             if ( (*it)->midPoint().z < zMin_ ) {
                (*it)->activeIs(false);
                (*it)->mesh()->showIs(false);
            }

        }
        i++;
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

void Engine::initParticleSystems() {

    std::cout << "initParticleSystems" << std::endl;

    std::string s1("../shaders/particle");
    std::string s2("../shaders/particle");
    std::string s3("../shaders/particle");
    std::string s4("../shaders/particle");
    std::string s5("../shaders/particle");
    std::string s6("../shaders/particle");
    std::string s7("../shaders/particle");
    std::string s8("../shaders/particle");
    std::string s9("../shaders/particle");

    fireEmitter1sd_ = new ShaderData(s1,true);
    fireEmitter2sd_ = new ShaderData(s2,true);
    debrisEmitter1sd_ = new ShaderData(s3,true);
    smokeEmittersd_ = new ShaderData(s4,true);
    missileSmokeEmittersd_ = new ShaderData(s5, true);
    missileFireEmittersd_ = new ShaderData(s6, true);
    debrisEmitter2sd_ = new ShaderData(s7, true);
    waterFoamEmitter1sd_ = new ShaderData(s8, true);
    waterFoamEmitter2sd_ = new ShaderData(s9, true);

    std::string t1("sprite");
    std::string t2("sprite");
    std::string t3("sprite");
    std::string t4("sprite");
    std::string t5("sprite");
    std::string t6("sprite");
    std::string t7("sprite");
    std::string t8("sprite");
    std::string t9("sprite");

    std::string p1("../textures/fire1.png");
    std::string p2("../textures/fire2.png");
    std::string p3("../textures/debris1.png");
    std::string p4("../textures/smoke.png");
    std::string p5("../textures/missileSmoke.png");
    std::string p6("../textures/missileFire.png");
    std::string p7("../textures/debris2.png");
    std::string p8("../textures/waterFoam1.png");
    std::string p9("../textures/waterFoam1.png");

    fireEmitter1sd_->enableMatrix(MODELVIEW);
    fireEmitter2sd_->enableMatrix(MODELVIEW);
    debrisEmitter1sd_->enableMatrix(MODELVIEW);
    debrisEmitter2sd_->enableMatrix(MODELVIEW);
    smokeEmittersd_->enableMatrix(MODELVIEW);
    missileSmokeEmittersd_->enableMatrix(MODELVIEW);
    missileFireEmittersd_->enableMatrix(MODELVIEW);
    waterFoamEmitter1sd_->enableMatrix(MODELVIEW);
    waterFoamEmitter2sd_->enableMatrix(MODELVIEW);

    fireEmitter1sd_->enableMatrix(PROJECTION);
    fireEmitter2sd_->enableMatrix(PROJECTION);
    debrisEmitter1sd_->enableMatrix(PROJECTION);
    debrisEmitter2sd_->enableMatrix(PROJECTION);
    smokeEmittersd_->enableMatrix(PROJECTION);
    missileSmokeEmittersd_->enableMatrix(PROJECTION);
    missileFireEmittersd_->enableMatrix(PROJECTION);
    waterFoamEmitter1sd_->enableMatrix(PROJECTION);
    waterFoamEmitter2sd_->enableMatrix(PROJECTION);

    Matrix4 * proj1 = fireEmitter1sd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj2 = fireEmitter2sd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj3 = debrisEmitter1sd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj4 = smokeEmittersd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj5 = missileSmokeEmittersd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj6 = missileFireEmittersd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj7 = debrisEmitter2sd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj8 = waterFoamEmitter1sd_->stdMatrix4Data(PROJECTION);
    Matrix4 * proj9 = waterFoamEmitter2sd_->stdMatrix4Data(PROJECTION);

    *proj1 = camera()->projectionMtx();
    *proj2 = camera()->projectionMtx();
    *proj3 = camera()->projectionMtx();
    *proj4 = camera()->projectionMtx();
    *proj5 = camera()->projectionMtx();
    *proj6 = camera()->projectionMtx();
    *proj7 = camera()->projectionMtx();
    *proj8 = camera()->projectionMtx();
    *proj9 = camera()->projectionMtx();

    fireEmitter1sd_->addTexture(t1,p1);
    fireEmitter2sd_->addTexture(t2,p2);
    debrisEmitter1sd_->addTexture(t3,p3);
    smokeEmittersd_->addTexture(t4,p4);
    missileSmokeEmittersd_->addTexture(t5,p5);
    missileFireEmittersd_->addTexture(t6,p6);
    debrisEmitter2sd_->addTexture(t7,p7);
    waterFoamEmitter1sd_->addTexture(t8, p8);
    waterFoamEmitter2sd_->addTexture(t9, p9);

    fireEmitter1sd_->addFloat("softDist", 0.15);
    fireEmitter2sd_->addFloat("softDist", 0.15);
    debrisEmitter1sd_->addFloat("softDist", 0.15);
    smokeEmittersd_->addFloat("softDist", 0.2);
    missileSmokeEmittersd_->addFloat("softDist", 0.2);
    missileFireEmittersd_->addFloat("softDist", 0.2);
    debrisEmitter2sd_->addFloat("softDist", 0.2);
    waterFoamEmitter1sd_->addFloat("softDist", 0.001);
    waterFoamEmitter2sd_->addFloat("softDist", 0.001);

    fireEmitter1sd_->addTexture("cocTex","CoC");
    fireEmitter2sd_->addTexture("cocTex","CoC");
    debrisEmitter1sd_->addTexture("cocTex","CoC");
    smokeEmittersd_->addTexture("cocTex","CoC");
    missileSmokeEmittersd_->addTexture("cocTex","CoC");
    missileFireEmittersd_->addTexture("cocTex","CoC");
    debrisEmitter2sd_->addTexture("cocTex","CoC");
    waterFoamEmitter1sd_->addTexture("cocTex","CoC");
    waterFoamEmitter2sd_->addTexture("cocTex","CoC");

    ParticleSystem * ps;
    ParticleSystem * ps2;

    // missile
    std::vector<MeshedProjectile*> missiles = rocketLauncher_->projectiles();
    std::vector<MeshedProjectile*>::iterator pit;
    for (pit=missiles.begin(); pit!=missiles.end(); pit++) {

        ps = new ParticleSystem(2);
        (*pit)->particleSystemIs(ps);

        Emitter* missileSmokeEmitter= ps->newEmitter(6,missileSmokeEmittersd_);
        missileSmokeEmitter->posIs((*pit)->position());
        missileSmokeEmitter->typeIs(Emitter::EMITTER_STREAM);
        missileSmokeEmitter->blendModeIs(Emitter::BLEND_SMOKE);
        missileSmokeEmitter->rateIs(0.08f);
        missileSmokeEmitter->lifeTimeIs(5.f);
        missileSmokeEmitter->massIs(1.f);
        missileSmokeEmitter->posRandWeightIs(0.f);
        missileSmokeEmitter->velIs(Vector3(0.f, 0.f, 0.f));
        missileSmokeEmitter->velRandWeightIs(1.f);
        missileSmokeEmitter->accIs(Vector3(0.f, 0.f, 0.0f));
        missileSmokeEmitter->pointSizeIs(1.f);
        missileSmokeEmitter->growthFactorIs(1.0f);

        Emitter* missileFireEmitter= ps->newEmitter(6,missileFireEmittersd_);
        missileFireEmitter->posIs((*pit)->position());
        missileFireEmitter->typeIs(Emitter::EMITTER_STREAM);
        missileFireEmitter->blendModeIs(Emitter::BLEND_FIRE);
        missileFireEmitter->rateIs(0.02f);
        missileFireEmitter->lifeTimeIs(4.f);
        missileFireEmitter->massIs(1.f);
        missileFireEmitter->posRandWeightIs(0.f);
        missileFireEmitter->velIs(Vector3(0.f, 0.f, 0.f));
        missileFireEmitter->velRandWeightIs(0.2f);
        missileFireEmitter->accIs(Vector3(0.f, 0.f, 0.0f));
        missileFireEmitter->pointSizeIs(0.5f);
        missileFireEmitter->growthFactorIs(1.0f);
    }

    // targets
    std::vector<Target*>::iterator it;
    for (it=targets_.begin(); it!=targets_.end(); it++) {

        ps = new ParticleSystem(5);

        (*it)->explosionPsIs(ps);

        Emitter * smokeEmitter = ps->newEmitter(10, smokeEmittersd_);
        smokeEmitter->posIs((*it)->midPoint());
        smokeEmitter->burstSizeIs(10);
        smokeEmitter->typeIs(Emitter::EMITTER_BURST);
        smokeEmitter->blendModeIs(Emitter::BLEND_SMOKE);
        smokeEmitter->rateIs(0.02f);
        smokeEmitter->lifeTimeIs(2.f);
        smokeEmitter->massIs(1.f);
        smokeEmitter->posRandWeightIs(0.2f);
        smokeEmitter->velIs(Vector3(0.f, 0.001f, 0.f));
        smokeEmitter->velRandWeightIs(0.001);
        smokeEmitter->accIs(Vector3(0.f, 0.0f, 0.0f));
        smokeEmitter->pointSizeIs(10.f);
        smokeEmitter->growthFactorIs(1.00f);

        Emitter * fireEmitter1 = ps->newEmitter(70, fireEmitter1sd_);
        fireEmitter1->posIs((*it)->midPoint());
        fireEmitter1->burstSizeIs(70);
        fireEmitter1->typeIs(Emitter::EMITTER_BURST);
        fireEmitter1->blendModeIs(Emitter::BLEND_FIRE);
        fireEmitter1->rateIs(0.02f);
        fireEmitter1->lifeTimeIs(1.6f);
        fireEmitter1->massIs(1.f);
        fireEmitter1->posRandWeightIs(1.2f);
        fireEmitter1->velIs(Vector3(-3.f, 0.f, 0.f));
        fireEmitter1->velRandWeightIs(2.f);
        fireEmitter1->accIs(Vector3(0.f, -5.f, 0.0f));
        fireEmitter1->pointSizeIs(2.f);
        fireEmitter1->growthFactorIs(1.0f);
        
        Emitter * fireEmitter2 = ps->newEmitter(70, fireEmitter2sd_);
        fireEmitter2->posIs((*it)->midPoint());
        fireEmitter2->burstSizeIs(70);
        fireEmitter2->typeIs(Emitter::EMITTER_BURST);
        fireEmitter2->blendModeIs(Emitter::BLEND_FIRE);
        fireEmitter2->rateIs(0.005f);
        fireEmitter2->lifeTimeIs(1.5f);
        fireEmitter2->massIs(1.f);
        fireEmitter2->posRandWeightIs(1.2f);
        fireEmitter2->velIs(Vector3(-3.f, 0.f, 0.f));
        fireEmitter2->velRandWeightIs(2.f);
        fireEmitter2->accIs(Vector3(0.f, -5.f, 0.0f));
        fireEmitter2->pointSizeIs(3.f);
        fireEmitter2->growthFactorIs(1.0f);


        Emitter * debrisEmitter1 = ps->newEmitter(20, debrisEmitter1sd_);
        debrisEmitter1->posIs((*it)->midPoint());
        debrisEmitter1->burstSizeIs(20);
        debrisEmitter1->typeIs(Emitter::EMITTER_BURST);
        debrisEmitter1->blendModeIs(Emitter::BLEND_SMOKE);
        debrisEmitter1->rateIs(0.02f);
        debrisEmitter1->lifeTimeIs(1.f);
        debrisEmitter1->massIs(1.f);
        debrisEmitter1->posRandWeightIs(0.1f);
        debrisEmitter1->velIs(Vector3(0.f, 2.f, 0.f));
        debrisEmitter1->velRandWeightIs(3.f);
        debrisEmitter1->accIs(Vector3(0.f, -20.f, 0.0f));
        debrisEmitter1->pointSizeIs(0.3f);
        debrisEmitter1->growthFactorIs(1.f);

        Emitter * debrisEmitter2 = ps->newEmitter(20, debrisEmitter2sd_);
        debrisEmitter2->posIs((*it)->midPoint());
        debrisEmitter2->burstSizeIs(20);
        debrisEmitter2->typeIs(Emitter::EMITTER_BURST);
        debrisEmitter2->blendModeIs(Emitter::BLEND_SMOKE);
        debrisEmitter2->rateIs(0.02f);
        debrisEmitter2->lifeTimeIs(1.f);
        debrisEmitter2->massIs(1.f);
        debrisEmitter2->posRandWeightIs(0.2f);
        debrisEmitter2->velIs(Vector3(0.f, 2.f, 0.f));
        debrisEmitter2->velRandWeightIs(4.f);
        debrisEmitter2->accIs(Vector3(0.f, -20.f, 0.0f));
        debrisEmitter2->pointSizeIs(0.4f);
        debrisEmitter2->growthFactorIs(1.f);

        ps2 = new ParticleSystem(2);
        (*it)->foamPsIs(ps2);

        Emitter * waterFoamLeft = ps2->newEmitter(30, waterFoamEmitter1sd_);
        waterFoamLeft->posIs((*it)->frontLeft());
        waterFoamLeft->typeIs(Emitter::EMITTER_STREAM);
        waterFoamLeft->blendModeIs(Emitter::BLEND_FIRE);
        waterFoamLeft->rateIs(0.01f);
        waterFoamLeft->lifeTimeIs(5.f);
        waterFoamLeft->massIs(1.f);
        waterFoamLeft->posRandWeightIs(0.2f);
        waterFoamLeft->velIs(Vector3(13.f, 0.f, 0.f));
        waterFoamLeft->velRandWeightIs(0.2f);
        waterFoamLeft->accIs(Vector3(-10.f, 0.f, 0.0f));
        waterFoamLeft->pointSizeIs(1.0f);

        waterFoamLeft->growthFactorIs(0.97f); 

        Emitter * waterFoamRight = ps2->newEmitter(30, waterFoamEmitter2sd_);
        waterFoamRight->posIs((*it)->frontRight());
        waterFoamRight->typeIs(Emitter::EMITTER_STREAM);
        waterFoamRight->blendModeIs(Emitter::BLEND_FIRE);
        waterFoamRight->rateIs(0.01f);
        waterFoamRight->lifeTimeIs(5.f);
        waterFoamRight->massIs(1.f);
        waterFoamRight->posRandWeightIs(0.2f);
        waterFoamRight->velIs(Vector3(-13.f, 0.f, 0.f));
        waterFoamRight->velRandWeightIs(0.2f);
        waterFoamRight->accIs(Vector3(10.f, 0.f, 0.0f));
        waterFoamRight->pointSizeIs(1.0f);
        waterFoamRight->growthFactorIs(0.97f);


    }
}

void Engine::displayParticles() {

    std::vector<MeshedProjectile*> missiles = rocketLauncher_->projectiles();
    std::vector<MeshedProjectile*>::iterator pit;
    for (pit=missiles.begin(); pit!=missiles.end(); pit++) {
        if ((*pit)->active() && (*pit)->flightDistance() > 15.f) {
            (*pit)->particleSystem()->display();
        }
    }

    std::vector<Target*>::iterator it;
    for (it=targets_.begin(); it!=targets_.end(); it++) {
        (*it)->explosionPs()->display();
        if ((*it)->active() ) {
            (*it)->foamPs()->display();
        }
    }
}

void Engine::UpdateDOF()
{
    Vector3 viewVector = activeCam_->viewVector();
    float t = (-activeCam_->worldPos(0.0f).y-5)/viewVector.y;
    focalPlane_ = t;
    nearBlurPlane_ = t-50.0f;
    if (nearBlurPlane_ < 0.0) nearBlurPlane_ = 0.0f;
    farBlurPlane_ = t+50.f;
    maxBlur_ = 0.8f;

    std::string focalPlaneStr("focalPlane");
    std::string nearPlaneStr("nearBlurPlane");
    std::string farPlaneStr("farBlurPlane");
    std::string maxStr("maxBlur");

    ShaderData* phong = shaders_["phongTarget"];
    float * phongFocal = phong->floatData(focalPlaneStr);
    float * phongNear = phong->floatData(nearPlaneStr);
    float * phongFar = phong->floatData(farPlaneStr);
    float * phongMax = phong->floatData(maxStr);

    ShaderData* ocean = shaders_["ocean"];
    float * oceanFocal = ocean->floatData(focalPlaneStr);
    float * oceanNear = ocean->floatData(nearPlaneStr);
    float * oceanFar = ocean->floatData(farPlaneStr);
    float * oceanMax = ocean->floatData(maxStr);

    *phongFocal = focalPlane_;
    *phongNear = nearBlurPlane_;
    *phongFar = farBlurPlane_;
    *phongMax = maxBlur_;

    *oceanFocal = focalPlane_;
    *oceanNear = nearBlurPlane_;
    *oceanFar = farBlurPlane_;
    *oceanMax = maxBlur_;
}

void Engine::updateParticles(float _dt) {

    std::vector<MeshedProjectile*> missiles = rocketLauncher_->projectiles();
    std::vector<MeshedProjectile*>::iterator pit;
    for (pit=missiles.begin(); pit!=missiles.end(); pit++) {
        (*pit)->particleSystem()->update(_dt);
    }
 
    std::vector<Target*>::iterator it;
    for (it=targets_.begin(); it!=targets_.end(); it++) {
        (*it)->explosionPs()->update(_dt);
        (*it)->foamPs()->update(_dt);
    }
}
