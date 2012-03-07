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

Graphics::Graphics()
{

}

Graphics::~Graphics()
{

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
    } else {
        std::cerr << "Buffers already exists for " << _name << std::endl;
    }
}

void Graphics::deleteBuffers(const std::string & _name)
{
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


void Graphics::bindGeometry(GLuint _VAO,
                            GLuint _VBO,
                            GLuint _n,
                            GLuint _stride,
                            GLuint _locIdx,
                            GLuint _offset)
{
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ARRAY_BUFFER, _VBO);

    glEnableVertexAttribArray(_locIdx);
    glVertexAttribPointer(_locIdx, _n, GL_FLOAT, 0, _stride,
            BUFFER_OFFSET(_offset));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void Graphics::drawIndices(GLuint _VAO,
                           GLuint _VBO,
                           GLuint _size,
                           const ShaderData * _shaderData)
{
    glBindVertexArray(_VAO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _VBO);

    glUseProgram(_shaderData->shaderID_);
    attribMap1f::const_iterator fIt = _shaderData->floats_.begin();
    attribMap3f::const_iterator vecIt = _shaderData->vec3s_.begin();
    attribMapMat4::const_iterator matIt = _shaderData->matrices_.begin();
    attribMapTex::const_iterator texIt = _shaderData->textures_.begin();

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
        glUniform1i(texIt->second.location, i);
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_2D, texIt->second.data);
    }

    glDrawElements(GL_TRIANGLES, _size, GL_UNSIGNED_INT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
    glBindVertexArray(0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
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
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                         GL_LINEAR_MIPMAP_NEAREST );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
        glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA, FI2T.w, FI2T.h, 0, GL_RGBA,
                GL_UNSIGNED_BYTE,FI2T.data );
        glBindTexture(GL_TEXTURE_2D, 0);
        texture_[_img] = texID;
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

    bool loaded;
    std::string error;
#define ERROR_BUFSIZE 1024
    // Error checking
    glGetProgramiv(programID, GL_LINK_STATUS, (GLint*)&loaded);
    //glGetShaderiv(vertexShaderID, GL_COMPILE_STATUS, (GLint*)&loaded_);
    if (!loaded) {
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

