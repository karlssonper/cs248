/*
 * Engine.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef ENGINE_H_
#define ENGINE_H_

class Engine
{
public:
    static Engine& instance() { static Engine e; return e; };
    void init(int argc, char **argv, const char * _name, int _width, int _height);
    void loadResources(const char * _file);
    void start();

    int mouseX() const { return mouseX_;};
    void mouseXIs(int x);
    int mouseY() const { return mouseY_;};
    void mouseYIs(int y);
private:
    int mouseX_;
    int mouseY_;

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
    unsigned int quadShader_;

    void BuildQuad();

    Engine();
    Engine(const Engine & );
    void operator=(const Engine & );
};


#endif /* ENGINE_H_ */
