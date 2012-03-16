/*
 * Engine.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef ENGINE_H_
#define ENGINE_H_

class ShaderData;
class Engine
{
public:
    static Engine& instance() { static Engine e; return e; };
    void init(int argc, char **argv, const char * _name, int _width, int _height);
    void loadResources(const char * _file);
    void start();

    void renderFrame();
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

    //Textures
    unsigned int shadowTex_;
    unsigned int phongTex_;
    unsigned int bloomTex_;
    unsigned int motionTex_;
    unsigned int cocTex_;

    //Framebuffers
    unsigned int shadowFB_;
    unsigned int firstPassFB_;
    unsigned int firstPassDepthFB_;
    unsigned int secondPassFB_;

    //Full screen texture quad
    struct QuadVertex { float pos[3]; float texCoords[2];};
    unsigned int quadVBO_;
    unsigned int quadIdxVBO_;
    unsigned int quadVAO_;
    ShaderData * quadShader_;

    void BuildQuad();

    void RenderShadowMap();
    void RenderFirstPass();
    void RenderSecondPass();


    Engine();
    Engine(const Engine & );
    void operator=(const Engine & );
};


#endif /* ENGINE_H_ */
