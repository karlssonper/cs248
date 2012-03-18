/*
 * Graphics.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Graphics.h"
#include <fstream>
#include <iostream>
#include "wrappers/FreeImage2Tex.h"
#include <CUDA.h>

Graphics::Graphics()
{
#ifdef USE_GLEW
    printf("OpenGL and GLEW");
    GLint error = glewInit();
    if (GLEW_OK != error) {
        std::cerr << glewGetErrorString(error) << std::endl;
        //exit(-1);
    }
    if (!GLEW_VERSION_3_2 || !GL_EXT_framebuffer_object) {
        std::cerr << "This program requires OpenGL 3.2" << std::endl;
        //exit(-1);
    }
#else
    printf("OpenGL and gl3w");
    if (gl3wInit()) {
        fprintf(stderr, "failed to initialize OpenGL\n");
        //exit(-1);
    }
    if (!gl3wIsSupported(3, 2)) {
        fprintf(stderr, "OpenGL 3.2 not supported\n");
        //exit(-1);
    }
    printf(" %s\nGLSL %s\n", glGetString(GL_VERSION),
           glGetString(GL_SHADING_LANGUAGE_VERSION));
#endif

    glEnable(GL_DEPTH_TEST);
    CUDA::init();

}

void Graphics::viewportIs(int _width, int _height)
{
    glViewport(0, 0, _width, _height);
}

void Graphics::buffersNew(const std::string & _name,
                GLuint & _VAO,
                GLuint & _geometryVBO,
                GLuint & _idxVBOO )
{
    if (VAOData_.find(_name) == VAOData_.end()) {
        glGenVertexArrays(1, &_VAO);
        glGenBuffers(1, &_geometryVBO);
        glGenBuffers(1, &_idxVBOO);
        //add to map
    } else {
        std::cerr << "Buffers already exists for " << _name << std::endl;
    }
}

void Graphics::buffersNew(const std::string &_name,
                          GLuint & _VAO,
                          GLuint & _positionVBO,
                          GLuint & _sizeVBO,
                          GLuint & _timeVBO)
{
    if (VAOData_.find(_name) == VAOData_.end()) {
        glGenVertexArrays(1, &_VAO);
        glGenBuffers(1, &_positionVBO);
        glGenBuffers(1, &_sizeVBO);
        glGenBuffers(1, &_timeVBO);
    } else {
        std::cerr << "Buffers already exists for " << _name << std::endl;
    }
}

void Graphics::deleteBuffers(const std::string & _name)
{
    std::cout << "deleteBuffers(" << std::cout << _name << ")" << std::endl;
    if (VAOData_.find(_name) != VAOData_.end()) {
        VAOData &S = VAOData_[_name];
        glDeleteBuffers(1, &S.geometryVBO);
        glDeleteBuffers(1, &S.indexVBO);
        glDeleteVertexArrays(1, &S.VAO);
        VAOData_.erase(_name);
    } else {
        std::cerr << "Can't remove buffers " << _name << std::endl;
    }
}

void Graphics::deleteBuffers(GLuint _VAO)
{
    for (VAOmap::iterator it = VAOData_.begin(); it != VAOData_.end(); ++it) {
        if (it->second.VAO == _VAO) {
            VAOData &S = VAOData_[it->first];
            glDeleteBuffers(1, &S.geometryVBO);
            glDeleteBuffers(1, &S.indexVBO);
            glDeleteVertexArrays(1, &S.VAO);
            VAOData_.erase(it);
            return;
        }
    }
    std::cerr << "Can't remove buffers  VAO#" << _VAO << std::endl;
}

void Graphics::createTextureToFBO(std::string _name,
                                  GLuint &_depthTex,
                                  GLuint &_fbo,
                                  GLuint _width,
                                  GLuint _height)
{
    GLuint depthTexture;
    glGenTextures(1, &depthTexture);

    glBindTexture(GL_TEXTURE_2D, depthTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            _width, _height,0,GL_DEPTH_COMPONENT,
            GL_UNSIGNED_BYTE,0);

    glBindTexture(GL_TEXTURE_2D, 0);

   // Generate a framebuffer
    glGenFramebuffers(1, &_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbo);

    // Attach the texture to the frame buffer
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_TEXTURE_2D, depthTexture, 0);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    FBO_[_name] = _fbo;
    texture_[_name] = depthTexture;
}

void Graphics::createTextureToFBO(const std::vector<std::string> &_names,
                                  std::vector<GLuint> &_colorTex,
                                  GLuint &_colorFBO,
                                  GLuint &_depthFBO,
                                  GLuint _width,
                                  GLuint _height)
{
    glGenFramebuffers(1, &_colorFBO);
    glGenRenderbuffers(1, &_depthFBO);

    glBindFramebuffer(GL_FRAMEBUFFER, _colorFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, _depthFBO);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _width, _height);

    FBO_[_names[0]] = _colorFBO;
    FBO_[_names[0]] = _depthFBO;

    for (unsigned int i = 0; i < _names.size(); ++i) {
        // The position buffer
        glActiveTexture(GL_TEXTURE0 + i); // Use texture unit 0 for position
        glGenTextures(1, &_colorTex[i]);
        texture_[_names[i]] = _colorTex[i];
        glBindTexture(GL_TEXTURE_2D, _colorTex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, _width, _height, 0,
                GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER, _depthFBO);
    for (unsigned int i = 0; i < _names.size(); ++i) {
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i,
        GL_TEXTURE_2D, _colorTex[i], 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Graphics::enableFramebuffer(GLuint _depthFBO,
                                 GLuint _colorFBO,
                                 GLuint _nDepth,
                                 GLuint _nColor,
                                 GLuint _width,
                                 GLuint _height)
{
    glBindFramebuffer(GL_FRAMEBUFFER, _colorFBO);
    glBindRenderbuffer(GL_RENDERBUFFER, _depthFBO);


    glViewport(0, 0, _width, _height);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    std::vector<GLenum> enums;
    _nDepth > 0 ? enums.push_back(GL_DEPTH_ATTACHMENT):enums.push_back(GL_NONE);

    for (unsigned int i = 0; i < _nColor; ++i) {
        enums.push_back(GL_COLOR_ATTACHMENT0 + i);
    }
    glDrawBuffers(enums.size(), &enums[0]);
}

void Graphics::enableFramebuffer(GLuint _depthFBO, GLuint _width,GLuint _height)
{
    glBindFramebuffer(GL_FRAMEBUFFER, _depthFBO);
    glDrawBuffer(GL_NONE);
    glCullFace(GL_FRONT);
    glViewport(0, 0, _width, _height);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Graphics::disableFramebuffer()
{
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
}

void Graphics::geometryIs(GLuint _posVBO,
                          GLuint _sizeVBO,
                          GLuint _timeVBO,
                          GLuint _N,
                          VBOtype _type)
{
    GLenum type;
    switch(_type){
        case VBO_STATIC:
            type = GL_STATIC_DRAW;
            break;
        case VBO_DYNAMIC:
            type = GL_DYNAMIC_DRAW;
            break;
    }

    std::vector<float> pos(3*_N);
    std::vector<float> size(_N);
    std::vector<float> time(_N);

    VBODataIs(GL_ARRAY_BUFFER, _posVBO, pos, type);
    VBODataIs(GL_ARRAY_BUFFER, _sizeVBO, size, type);
    VBODataIs(GL_ARRAY_BUFFER, _timeVBO, time, type);
}


void Graphics::bindGeometry(GLuint _shader,
                            GLuint _VAO,
                            GLuint _VBO,
                            GLuint _n,
                            GLuint _stride,
                            GLuint _locIdx,
                            GLuint _offset)
{
    glUseProgram(_shader);
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);

    glEnableVertexAttribArray(_locIdx);
    glVertexAttribPointer(_locIdx, _n, GL_FLOAT, 0, _stride,
            BUFFER_OFFSET(_offset));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glUseProgram(0);
}

void Graphics::drawArrays(GLuint _VAO,
                 GLuint _N,
                 const ShaderData * _shaderData,
                 bool additiveBlending)
{
    //glEnable(GL_POINT_SIZE);
    // draw
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);

    if (additiveBlending) {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else  {
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    loadShaderData(_shaderData);
    glBindVertexArray(_VAO);
    glDrawArrays(GL_POINTS, 0, _N);
    glDepthMask(GL_TRUE);
    glBindVertexArray(0);
    unloadShaderData();
}

void Graphics::drawIndices(GLuint _VAO,
                           GLuint _VBO,
                           GLuint _size,
                           const ShaderData * _shaderData)
{
    glUseProgram(_shaderData->shaderID());
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _VBO);

    loadShaderData(_shaderData);

    glDrawElements(GL_TRIANGLES, _size, GL_UNSIGNED_INT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void Graphics::loadShaderData(const ShaderData * _shaderData) const
{
    glUseProgram(_shaderData->shaderID());
    attribMap1f::const_iterator fIt = _shaderData->floats_.begin();
    attribMap3f::const_iterator vecIt = _shaderData->vec3s_.begin();
    attribMapMat4::const_iterator matIt = _shaderData->matrices_.begin();
    attribMapTex::const_iterator texIt = _shaderData->textures_.begin();
    attribMapTex::const_iterator ctexIt = _shaderData->cubeTextures_.begin();

    for (; fIt != _shaderData->floats_.end(); ++fIt){
       glUniform1f(fIt->second.location, fIt->second.data);
    }

    for (; vecIt != _shaderData->vec3s_.end(); ++vecIt){
       glUniform3fv(vecIt->second.location, 1, &vecIt->second.data.x);
    }

    for (; matIt != _shaderData->matrices_.end(); ++matIt){
       glUniformMatrix4fv(matIt->second.location, 1, false,
               matIt->second.data.data());
    }

    for (int i = 0; texIt != _shaderData->textures_.end(); ++texIt, ++i){
       glUniform1i(texIt->second.location, i+1);
       glActiveTexture(GL_TEXTURE0 + i+1);
       glBindTexture(GL_TEXTURE_2D, texIt->second.data);
    }

    for (int i = 0; ctexIt != _shaderData->cubeTextures_.end(); ++ctexIt, ++i){
       glUniform1i(ctexIt->second.location, i);
       glActiveTexture(GL_TEXTURE0 + i);
       glBindTexture(GL_TEXTURE_CUBE_MAP, ctexIt->second.data);
    }

    for (int i = 0; i < NUM_STD_MATRICES-1; ++i) {
       if (_shaderData->stdMatrices_[i].first) {
           glUniformMatrix4fv(_shaderData->stdMatrices_[i].second.location,
                   1,
                   false,
                   _shaderData->stdMatrices_[i].second.data.data());
       }
    }
    if (_shaderData->stdMatrixNormal_.first) {
       glUniformMatrix3fv(_shaderData->stdMatrixNormal_.second.location,
                   1,
                   false,
                   _shaderData->stdMatrixNormal_.second.data.data());
    }
}

void Graphics::unloadShaderData() const
{
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

GLuint Graphics::texture(const std::string & _img)
{
    if (texture_.find(_img) != texture_.end()) {
        return texture_[_img];
    } else {
        FreeImage2Texture FI2T(_img);
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_2D, texID);
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA, FI2T.w, FI2T.h, 0, GL_RGBA,
                GL_UNSIGNED_BYTE,FI2T.data);
        glGenerateMipmap(GL_TEXTURE_2D);
        //glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                //GL_LINEAR_MIPMAP_NEAREST );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );

        glBindTexture(GL_TEXTURE_2D, 0);
        texture_[_img] = texID;
        return texID;
    }
}

GLuint Graphics::texture(const std::string & _name,
                         const std::vector<std::string> &_img)
{
    if (texture_.find(_name) != texture_.end()) {
        return texture_[_name];
    } else {
        if (_img.size() != 6) {
            std::cerr << "Error! A cubemap needs 6 textures" << std::endl;
        }
        GLuint texID;
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_CUBE_MAP, texID);
        glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameterf(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);
        glTexParameterf(GL_TEXTURE_CUBE_MAP,GL_TEXTURE_WRAP_R,GL_CLAMP_TO_EDGE);
        for (unsigned int face = 0; face < 6; face++){
            FreeImage2Texture FI2T(_img[face]);
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face, 0, GL_RGBA,
                    FI2T.w, FI2T.h, 0 , GL_RGBA, GL_UNSIGNED_BYTE, FI2T.data);
            //glCopyTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + face,
            //        0, 0, 0, 0, 0, FI2T.w, FI2T.h);
        }
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
        texture_[_name] = texID;
        return texID;
    }
}

void Graphics::deleteTexture(const std::string & _img)
{
    if (texture_.find(_img) != texture_.end()) {
        glDeleteTextures(1, &texture_[_img]);
    } else {
        std::cerr << "Can't remove texture " << _img << std::endl;
    }
}

void Graphics::deleteTexture(unsigned int _texID)
{
    for (TexMap::iterator it = texture_.begin(); it != texture_.end(); ++it) {
        if (it->second == _texID) {
            glDeleteTextures(1, &it->second);
            texture_.erase(it);
            return;
        }
    }
    std::cerr << "Can't remove texture #" << _texID << std::endl;
}

GLuint Graphics::shader(const std::string & _shader)
{
    if(shader_.find(_shader) == shader_.end()){
        return LoadShader(_shader);
    } else {
        return shader_[_shader].programID;
    }
}

GLuint Graphics::LoadShader(const std::string _shader)
{
    const GLchar* source[1];
    int length = 0;

    // Load the fragment shader and compile
    std::vector<char> fragmentSource = ReadSource(_shader + ".frag.glsl");
    source[0] = &fragmentSource.front();
    length = fragmentSource.size()-1;
    GLuint fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShaderID, 1, source, &length);
    glCompileShader(fragmentShaderID);

    // Load the vertex shader and compile
    std::vector<char> vertexSource = ReadSource(_shader + ".vert.glsl");
    source[0] = &vertexSource.front();
    length = vertexSource.size()-1;
    GLuint vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShaderID, 1, source, &length);
    glCompileShader(vertexShaderID);

    // Create the vertex program
    GLuint programID = glCreateProgram();
    glAttachShader(programID, fragmentShaderID);
    glAttachShader(programID, vertexShaderID);
    glLinkProgram(programID);

   // bool loaded;
    GLint loaded = 0;
    std::string error;
#define ERROR_BUFSIZE 1024
    // Error checking
    glGetProgramiv(programID, GL_LINK_STATUS, &loaded);
    //glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, (GLint*)&loaded_);
    if (loaded == 0) {
        GLchar tempErrorLog[ERROR_BUFSIZE];
        GLsizei length;
        glGetShaderInfoLog(fragmentShaderID,ERROR_BUFSIZE,&length,tempErrorLog);
        error += "Fragment shader errors:\n";
        error += std::string(tempErrorLog, length) + "\n";
        glGetShaderInfoLog(vertexShaderID, ERROR_BUFSIZE, &length,tempErrorLog);
        error += "Vertex shader errors:\n";
        error += std::string(tempErrorLog, length) + "\n";
        glGetProgramInfoLog(programID, ERROR_BUFSIZE, &length, tempErrorLog);
        error += "Linker errors:\n";
        error += std::string(tempErrorLog, length) + "\n";
        std::cerr << error;
    }

    ShaderID &S = shader_[_shader];
    S.vertexShaderId = vertexShaderID;
    S.fragmentShaderID = fragmentShaderID;
    S.programID = programID;

    return programID;
}

std::vector<char> Graphics::ReadSource(const std::string _file)
{

    // Open the file
    std::vector<char> source;
    std::ifstream in(_file.c_str());
    if (in.fail()) {
        source.push_back(0);
        return source;
    }

    // Seek to the end of the file to get the size
    in.seekg(0, std::ios::end);
    source.reserve((unsigned)(1 + in.tellg()));
    source.resize((unsigned)in.tellg());
    in.seekg(0, std::ios::beg);
    if (source.empty()) {
        source.push_back(0);
        return source;
    }

    // Now read the whole buffer in one call, and don't
    // forget to null-terminate the vector with a zero
    in.read(&source.front(), source.size());
    source.push_back(0);

    return source;
}

void Graphics::deleteShader(const std::string & _shader)
{
    if (shader_.find(_shader) != shader_.end()) {
        ShaderID &S = shader_[_shader];
        glDeleteShader(S.vertexShaderId);
        glDeleteShader(S.fragmentShaderID);
        glDeleteProgram(S.programID);
        shader_.erase(_shader);
    } else {
        std::cerr << "Can't remove shader " << _shader << std::endl;
    }
}

void Graphics::deleteShader(unsigned int _shaderID)
{
    for (ShaderMap::iterator it = shader_.begin(); it != shader_.end(); ++it) {
        if (it->second.programID == _shaderID) {
            ShaderID &S = shader_[it->first];
            glDeleteShader(S.vertexShaderId);
            glDeleteShader(S.fragmentShaderID);
            glDeleteProgram(S.programID);
            shader_.erase(it);
            return;
        }
    }
    std::cerr << "Can't remove shader #" << _shaderID << std::endl;
}

GLint Graphics::shaderUniformLoc(GLuint _shader, const std::string & _name)
{
    GLint loc = glGetUniformLocation(_shader, _name.c_str());
    if (loc < 0){
        //std::cerr << "Couldn't find location for " << _name << std::endl;
    }
    return loc;
}

GLint Graphics::shaderAttribLoc(GLuint _shader, const std::string & _name)
{
    GLint loc = glGetAttribLocation(_shader, _name.c_str());
    if (loc < 0){
            //std::cerr << "Couldn't find location for " << _name << std::endl;
        }
    return loc;
}

void Graphics::cleanUp() {
    VAOData_.clear();
    texture_.clear();
    FBO_.clear();
    shader_.clear();
}

Graphics::~Graphics() {
    std::cout << "~Graphics()" << std::endl;

}
