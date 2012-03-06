#include "ShaderProgram.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <GL/glew.h>
#include <GL/glut.h>

ShaderProgram::ShaderProgram() {
    program_ = glCreateProgram();
}

std::string ShaderProgram::readShaderTextFile(const std::string &_fileName) {
    std::ifstream in(_fileName.c_str());
    if (!in) {
        std::cerr << "Failed to read '" << _fileName << "'" << std::endl;
        exit(1);
    }
    std::stringstream ss;
    ss << in.rdbuf();
    std::string str = ss.str();
    return str;
}

void ShaderProgram::loadVertexShader(const std::string &_fileName) {
    const char* source = readShaderTextFile(_fileName).c_str();
    std::cout << source << std::endl;
    GLuint shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint result = 0;
    static const int kBufferSize = 1024;
    char buffer[1024];

    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if(result != GL_TRUE) {
        GLsizei length = 0;
        glGetShaderInfoLog(shader, kBufferSize-1,
            &length, buffer);
        std::cerr << _fileName << " GLSL error\n" << buffer << std::endl;
        exit(1);
    }

    glAttachShader(program_, shader);
    glLinkProgram(program_);
}

void ShaderProgram::loadFragmentShader(const std::string &_fileName) {
    const char* source = readShaderTextFile(_fileName).c_str();
    std::cout << source << std::endl;
    GLuint shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    GLint result = 0;
    static const int kBufferSize = 1024;
    char buffer[1024];
 
    glGetShaderiv(shader, GL_COMPILE_STATUS, &result);
    if(result != GL_TRUE) {
        GLsizei length = 0;
        glGetShaderInfoLog(shader, kBufferSize-1,
            &length, buffer);
        std::cerr << _fileName << " GLSL error\n" << buffer << std::endl;
        exit(1);
    }

    glAttachShader(program_, shader);
    glLinkProgram(program_);
}

void ShaderProgram::bind() {
    glUseProgram(program_);
}

void ShaderProgram::unBind() {
    glUseProgram(0);
}

void ShaderProgram::setUniform1f(const std::string &_uniform, float value) {
    const char* uniform = _uniform.c_str();
    glUniform1f(glGetUniformLocation(program_, uniform), value);
}