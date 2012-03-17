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

class Node;
class Mesh;
class ShaderData;
class Camera;
class Engine
{
public:
    static Engine& instance() { static Engine e; return e; };
    void init(int argc, char **argv, const char * _name, int _width, int _height);
    void loadResources(const char * _file);
    void start();

    void renderFrame(float _currentTime);
    Camera * camera() const { return activeCam_;};

    int mouseX() const { return mouseX_;};
    void mouseXIs(int x);
    int mouseY() const { return mouseY_;};
    void mouseYIs(int y);
    int width() const { return width_;};
    void widthIs(int _width);
    int height() const { return height_;};
    void heightIs(int _height);

    void cleanUp();
private:
    ~Engine();
    int mouseX_;
    int mouseY_;
    int width_;
    int height_;

    enum State { NOT_INITIATED, RUNNING, PAUSED};
    State state_;
    float currentTime_;
    Node * root_;

    Camera * activeCam_;
    Camera * gameCam_;
    Camera * freeCam_;
    Camera * lightCam_;

    typedef std::map<std::string, Mesh*> MeshMap;
    typedef std::map<std::string, Node*> NodeMap;
    typedef std::map<std::string, ShaderData*> ShaderMap;
    MeshMap meshes_;
    NodeMap nodes_;
    ShaderMap shaders_;

    //Textures
    unsigned int phongTex_;
    unsigned int bloomTex_;
    unsigned int motionTex_;
    unsigned int cocTex_;

    //Framebuffers
    unsigned int firstPassFB_;
    unsigned int firstPassDepthFB_;
    unsigned int secondPassFB_;

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

    void RenderShadowMap();
    void RenderFirstPass();
    void RenderSecondPass();

    Engine();
    Engine(const Engine & );
    void operator=(const Engine & );
};


#endif /* ENGINE_H_ */
