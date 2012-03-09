/*
 * ShaderData.cpp
 *
 *  Created on: Mar 6, 2012
 *      Author: per
 */

#include "ShaderData.h"
#include "Graphics.h"

static std::string STDMatricesStr[NUM_STD_MATRICES] = {
        "ModelViewMatrix",
        "ProjectionMatrix",
        "NormalMatrix",
        "ModelMatrix",
        "LightViewMatrix",
        "LightProjectionMatrix",
        "InverseViewMatrix"
};

ShaderData::ShaderData(const std::string & _shader) : shaderName_(_shader)
{
    shaderID_ = Graphics::instance().shader(_shader);
    stdMatrices_.resize(NUM_STD_MATRICES);
    for (int i = 0; i < NUM_STD_MATRICES; ++i){
        stdMatrices_[i].first = false;
        stdMatrices_[i].second.location =
            Graphics::instance().shaderUniformLoc(shaderID_, STDMatricesStr[i]);
    }
}

void ShaderData::enableMatrix(STD_Matrix _m)
{
    stdMatrices_[_m].first = true;
}

void ShaderData::disableMatrix(STD_Matrix _m)
{
    stdMatrices_[_m].first = false;
}

void ShaderData::addFloat(const std::string & _name, float _value)
{
    ShaderAttribute<float> &V = floats_[_name];
    V.data = _value;
    V.location = Graphics::instance().shaderUniformLoc(shaderID_, _name);
}

void ShaderData::addVector3(const std::string & _name, const Vector3 &_vec)
{
    ShaderAttribute<Vector3> &V = vec3s_[_name];
    V.data = _vec;
    V.location = Graphics::instance().shaderUniformLoc(shaderID_, _name);
}

void ShaderData::addMatrix(const std::string & _name, const Matrix4 & _m)
{
    ShaderAttribute<Matrix4> &V = matrices_[_name];
    V.data = _m;
    V.location = Graphics::instance().shaderUniformLoc(shaderID_, _name);
}

void ShaderData::addTexture(const std::string & _name, unsigned int _tex)
{
    ShaderAttribute<unsigned int> &V = textures_[_name];
    V.data = _tex;
    V.location = Graphics::instance().shaderUniformLoc(shaderID_, _name);
}

float * ShaderData::floatData(const std::string & _name)
{
    if (floats_.find(_name) != floats_.end()){
        return &floats_[_name].data;
    } else {
        std::cerr << "Can't find float Shader Attribute named " <<
                _name << std::endl;
    }

}

Vector3 * ShaderData::vector3Data(const std::string & _name)
{
    if (vec3s_.find(_name) != vec3s_.end()){
        return &vec3s_[_name].data;
    } else {
        std::cerr << "Can't find Vector3 Shader Attribute named " <<
                _name << std::endl;
    }
}

Matrix4 * ShaderData::matrixData(const std::string & _name)
{
    if (matrices_.find(_name) != matrices_.end()){
        return &matrices_[_name].data;
    } else {
        std::cerr << "Can't find Matrix Shader Attribute named " <<
                _name << std::endl;
    }
}

Matrix4 * ShaderData::stdMatrixData(STD_Matrix _m)
{
    return &stdMatrices_[_m].second.data;
}

unsigned int * ShaderData::textureData(const std::string & _name)
{
    if (textures_.find(_name) != textures_.end()){
        return &textures_[_name].data;
    } else {
        std::cerr << "Can't find Texture Shader Attribute named " <<
                _name << std::endl;
    }
}
