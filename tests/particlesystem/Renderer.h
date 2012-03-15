#ifndef RENDERER_H
#define RENDERER_H

#include <ParticleSystem.h>
#include "Shader.h"

#include <GL/glew.h>
#include <GL/glut.h>

#ifdef _WIN32
#include <GL/FreeImage.h>
#else
#include <FreeImage.h>
#endif

class Renderer {
public:
    Renderer(ParticleSystem *_ps, Shader *shader);
    void render();
    GLuint loadTexture(std::string _source);
private:
    ParticleSystem *ps_;
    Shader *shader_;
};

#endif
