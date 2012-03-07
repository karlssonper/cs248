#ifndef RENDERER_H
#define RENDERER_H

#include "../ParticleSystem.h"
#include "Shader.h"

#include <GL/glew.h>
#include <GL/glut.h>

class Renderer {
public:
    Renderer(ParticleSystem *_ps, Shader *shader);
    void render();
private:
    ParticleSystem *ps_;
    Shader *shader_;
};

#endif