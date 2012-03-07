#ifndef RENDERER_H
#define RENDERER_H

#include "../ParticleSystem.h"
#include "Shader.h"

#include <GL/glew.h>
#include <GL/glut.h>

#include <GL/FreeImage.h>

class Renderer {
public:
    Renderer(ParticleSystem *_ps, Shader *shader);
    void render();
    void loadTexture(std::string _source);
private:
    ParticleSystem *ps_;
    Shader *shader_;
    GLuint texture;
};

#endif