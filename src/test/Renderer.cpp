#include "Renderer.h"

Renderer::Renderer(ParticleSystem *_ps, Shader *_shader) 
    : ps_(_ps), shader_(_shader) {}

void Renderer::render() {

    for (unsigned int i=0; i<ps_->numEmitters(); ++i) {

        Emitter* emitter = ps_->emitter(i);

        // bind shader
        glUseProgram(shader_->programID());

        // get attribute and uniform handles
        GLint position = glGetAttribLocation(shader_->programID(), "positionIn");
        GLint color = glGetAttribLocation(shader_->programID(), "colorIn");
        GLint time = glGetAttribLocation(shader_->programID(), "timeIn");
        GLint sprite = glGetUniformLocation(shader_->programID(), "sprite");

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
        
        glEnable(GL_POINT_SIZE);
        glEnable(GL_TEXTURE_2D);

        // bind texture
        glUniform1i(sprite, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);

        // draw
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);

        glDrawArrays(GL_POINTS, 0, emitter->params().numParticles_);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        glUseProgram(0);

    }

}

void Renderer::loadTexture(std::string _source) {

    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(_source.c_str(), 0);
    FIBITMAP *image = FreeImage_Load(format, _source.c_str());

    int w = FreeImage_GetWidth(image);
    int h = FreeImage_GetHeight(image);

    std::cout << "Loaded image: " << w << " x " << w << std::endl;

    GLubyte* data = new GLubyte[4*w*h];
    data = (GLubyte*)FreeImage_GetBits(image);

    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, w, h, GL_BGRA, GL_UNSIGNED_BYTE, data);


}

