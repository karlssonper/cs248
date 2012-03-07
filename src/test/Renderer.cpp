#include "Renderer.h"

Renderer::Renderer(ParticleSystem *_ps, Shader *_shader) 
    : ps_(_ps), shader_(_shader) {}

void Renderer::render() {

    for (unsigned int i=0; i<ps_->numEmitters(); ++i) {

        Emitter* emitter = ps_->emitter(i);

        // bind shader
        glUseProgram(shader_->programID());

        // get attribute handles
        GLint position = glGetAttribLocation(shader_->programID(), "positionIn");
        GLint color = glGetAttribLocation(shader_->programID(), "colorIn");
        GLint time = glGetAttribLocation(shader_->programID(), "timeIn");

        // enable vertex attribute arrays
        glEnableVertexAttribArray(position);
        glEnableVertexAttribArray(color);
        glEnableVertexAttribArray(time);

        // bind and set array pointers
        glBindBuffer(GL_ARRAY_BUFFER, emitter->vboPos());
        glVertexAttribPointer(position,3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, emitter->vboCol());
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindBuffer(GL_ARRAY_BUFFER, emitter->vboTime());
        glVertexAttribPointer(time, 1, GL_FLOAT, GL_FALSE, 0, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // draw
        glEnable(GL_POINT_SIZE);
        glPointSize(10.f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_TRUE);

        glDrawArrays(GL_POINTS, 0, emitter->params().numParticles_);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glUseProgram(0);

    }





}