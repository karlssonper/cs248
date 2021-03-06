/*
 * Engine.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef ENGINE_H_
#define ENGINE_H_

#include <map>
#include <string>
#include <vector>

class Node;
class Mesh;
class ShaderData;
class Camera;
class Target;
class MeshedWeapon;
class Engine
{
public:
    static Engine& instance() { static Engine e; return e; };
    void init(int argc, char **argv, const char * _name, int _width, int _height);
    void loadResources(const char * _file);
    void start();

    void renderFrame(float _currentTime);
    void renderTexture(float v);
    void changeCamera();
    void toggleDOF();
    void toggleHUD();
    Camera * camera() const { return activeCam_;};
    Camera * lightCamera() const { return lightCam_;};

    int mouseX() const { return mouseX_;};
    void mouseXIs(int x);
    int mouseY() const { return mouseY_;};
    void mouseYIs(int y);
    int width() const { return width_;};
    void widthIs(int _width);
    int height() const { return height_;};
    void heightIs(int _height);

    std::vector<Target*> targets() const { return targets_; }

    unsigned int nrTargets() const { return nrTargets_; }
    void nrTargetsIs(unsigned int _nrTargets);
    float targetSpawnRate() const { return targetSpawnRate_; }
    void targetSpawnRateIs(float _targetSpawnRate);

    void xzBoundsIs(float _xMin, float _xMax, float _zMin, float _zMax);

    Node * root() const { return root_; }

    MeshedWeapon * rocketLauncher() const { return rocketLauncher_; }

    void cleanUp();
private:
    ~Engine();
    int mouseX_;
    int mouseY_;
    int width_;
    int height_;

    bool useDOF_;
    bool useHUD_;

    enum State { NOT_INITIATED, RUNNING, PAUSED};
    State state_;
    float currentTime_;
    Node * root_;

    Camera * activeCam_;
    bool updateCamView_;
    Camera * gameCam_;
    Camera * freeCam_;
    Camera * lightCam_;

    typedef std::map<std::string, Mesh*> MeshMap;
    typedef std::map<std::string, Node*> NodeMap;
    typedef std::map<std::string, ShaderData*> ShaderMap;
    MeshMap meshes_;
    NodeMap nodes_;
    ShaderMap shaders_;

    std::vector<Target*> targets_;

    float xMax_;
    float xMin_;
    float zMax_;
    float zMin_;
    
    //Depth of field
    float focalPlane_;
    float nearBlurPlane_;
    float farBlurPlane_;
    float maxBlur_;

    //Textures
    unsigned int depthTex_;
    unsigned int phongTex_;
    unsigned int bloomTex_;
    unsigned int bloom2Tex_;
    unsigned int motionTex_;
    unsigned int cocTex_;
    unsigned int coc2Tex_;

    //Framebuffers
    unsigned int firstPassFB_;
    unsigned int firstPassDepthFB_;
    unsigned int secondPassFB_;
    unsigned int secondPassDepthFB_;
    unsigned int horBlurFB_;
    unsigned int horBlurDepthFB_;
    unsigned int softParticlesFB_;
    unsigned int softParticlesDepthFB_;
    ShaderData * horizontalGaussianShader_;
    ShaderData * horDOFShader_;
    ShaderData * vertDOFShader_;

    //Shadow mapping
    unsigned int shadowFB_;
    unsigned int shadowTex_;
    unsigned int shadowSize_;
    ShaderData * shadowShader_;

    //Full screen texture quad
    struct QuadVertex { float pos[3]; float texCoords[2];};
    unsigned int quadVBO_;
    unsigned int quadIdxVBO_;
    unsigned int quadVAO_;
    ShaderData * quadShader_;

    //Skybox
    struct SkyboxVertex{ float pos[3];};
    unsigned int skyboxVBO_;
    unsigned int skyboxIdxVBO_;
    unsigned int skyboxVAO_;
    ShaderData * skyBoxShader_;

    void BuildQuad();
    void BuildSkybox();

    void BlurTextures();
    void RenderShadowMap();
    void RenderFirstPass();
    void RenderSecondPass();

    Engine();
    Engine(const Engine & );
    void operator=(const Engine & );

    unsigned int nrTargets_;
    float targetSpawnRate_;
    float nextSpawn_;

    void CreateFramebuffer();
    void LoadCameras();
    void LoadLight();
    void LoadOcean();
    void LoadTargets();
    void UpdateTargets(float _frameTime);
    void SpawnTargets();
    void UpdateDOF();
    void ScatterTargets();

    void initParticleSystems();
    void displayParticles();
    void updateParticles(float _dt);

    void loadWeapons();
    void updateProjectiles(float _dt);

    ShaderData * fireEmitter1sd_;
    ShaderData * fireEmitter2sd_;
    ShaderData * smokeEmittersd_;
    ShaderData * debrisEmitter1sd_;
    ShaderData * debrisEmitter2sd_;
    ShaderData * missileSmokeEmittersd_;
    ShaderData * missileFireEmittersd_;
    ShaderData * waterFoamEmitter1sd_;
    ShaderData * waterFoamEmitter2sd_;
    ShaderData * splashEmitter1sd_;
    ShaderData * splashEmitter2sd_;
    MeshedWeapon * rocketLauncher_;

    bool scatter;
};


#endif /* ENGINE_H_ */
