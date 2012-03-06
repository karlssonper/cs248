#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <string>

class ShaderProgram {
public:
    ShaderProgram();
    void loadVertexShader(const std::string &_fileName);
    void loadFragmentShader(const std::string &_fileName);
    void bind();
    void unBind();
    void setUniform1f(const std::string &_uniform, float value);
private:
    ShaderProgram(const ShaderProgram&);
    std::string readShaderTextFile(const std::string &_fileName);
    unsigned int program_;

};

#endif