#ifndef RENDERER_H
#define RENDERER_H

#include "Shader.h"

#include <GL/glew.h>
#include <GL/glut.h>

class Renderer {
public:
    Renderer(unsigned int _vbo, Shader *shader);
    void render();
private:
    unsigned int vbo_;
    Shader *shader_;
};

#endif