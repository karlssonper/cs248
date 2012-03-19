/*
 * ShaderData.h
 *
 *  Created on: Mar 6, 2012
 *      Author: per
 */

#ifndef SHADERDATA_H_
#define SHADERDATA_H_

#include <map>
#include <vector>
#include "MathEngine.h"


enum STD_Matrix {
    MODELVIEW             = 0,
    PROJECTION            = 1,
    MODEL                 = 2,
    LIGHTVIEW             = 3,
    LIGHTPROJECTION       = 4,
    INVERSEVIEW           = 5,
    //INVERSEVIEWPROJECTION = 6,
    //PREVVIEWPROJETION     = 7,
    NORMAL                = 6,
    NUM_STD_MATRICES      = 7
};

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
    ShaderData(const std::string & _shader, bool geomShader = false);

    unsigned int shaderID() const { return shaderID_; } ;
    void addFloat(const std::string & _name, float _value);
    void addVector3(const std::string & _name, const Vector3 &_vec);
    void addMatrix(const std::string & _name, const Matrix4 & _m);
    void addTexture(const std::string & _name, const std::string& _tex);
    void addCubeTexture(const std::string & _name,
                        const std::string & _texNameID,
                        const std::vector<std::string> &_img);

    void enableMatrix(STD_Matrix _m);
    void disableMatrix(STD_Matrix _m);

    float * floatData(const std::string & _name);
    Vector3 * vector3Data(const std::string & _name);
    Matrix4 * matrixData(const std::string & _name);
    Matrix4 * stdMatrix4Data(STD_Matrix _m);
    Matrix3 * stdMatrix3Data(STD_Matrix _m);
    unsigned int * textureData(const std::string & _name);

private:
    friend class Graphics;

    std::string shaderName_;
    unsigned int shaderID_;

    typedef std::vector<std::pair<bool, ShaderAttribute <Matrix4> > > matrixVec;

    matrixVec stdMatrices_;
    std::pair<bool, ShaderAttribute <Matrix3> > stdMatrixNormal_;
    attribMap1f floats_;
    attribMap3f vec3s_;
    attribMapMat4 matrices_;
    attribMapTex textures_;
    attribMapTex cubeTextures_;

};


#endif /* SHADERDATA_H_ */
