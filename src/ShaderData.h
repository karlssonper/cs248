/*
 * ShaderData.h
 *
 *  Created on: Mar 6, 2012
 *      Author: per
 */

#ifndef SHADERDATA_H_
#define SHADERDATA_H_

#include <map>
#include "MathEngine.h"

template <class T>
struct ShaderAttribute
{
    unsigned int location;
    T data;
};
typedef std::map<std::string, ShaderAttribute<float> > attribMap1f;
typedef std::map<std::string, ShaderAttribute<Vector3> > attribMap3f;
typedef std::map<std::string, ShaderAttribute<Matrix4> > attribMapMat4;
typedef std::map<std::string, ShaderAttribute<unsigned int> > attribMapTex;

class ShaderData
{
public:
    ShaderData(const std::string & _shader);
    ShaderData(unsigned int _shader);

    void addFloat(const std::string & _name, float _value);
    void addVector3(const std::string & _name, const Vector3 &_vec);
    void addMatrix(const std::string & _name, const Matrix4 & _m);
    void addTexture(const std::string & _name, unsigned int _tex);

    float * floatData(const std::string & _name);
    Vector3 * vector3Data(const std::string & _name);
    Matrix4 * matrixData(const std::string & _name);
    unsigned int * textureData(const std::string & _name);

private:
    friend class Graphics;

    std::string shaderName_;
    unsigned int shaderID_;

    attribMap1f floats_;
    attribMap3f vec3s_;
    attribMapMat4 matrices_;
    attribMapTex textures_;

};


#endif /* SHADERDATA_H_ */
