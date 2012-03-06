#include "Renderer.h"

Renderer::Renderer(GLuint _vbo, Shader *_shader) 
    : vbo_(_vbo), shader_(_shader) {}

void Renderer::render() {
    glUseProgram(shader_->programID());
    glEnableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexPointer(3, GL_FLOAT, 0, 0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glUseProgram(0);
}