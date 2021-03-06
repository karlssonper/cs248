/*
 * Mesh.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Mesh.h"
#include "Node.h"
#include "ShaderData.h"
#include "Graphics.h"
#include "Engine.h"
#include "Camera.h"
#include <limits>

enum SHADER_ATTRIBUTES_IDX {
    POSITIONS = 0,
    TEXCOORDS = 1,
    NORMALS = 2,
    TANGENTS = 3,
    BITANGENTS = 4,
    NUM_SHADER_ATTRIBUTES = 5
};

static std::string ShaderAttributes[NUM_SHADER_ATTRIBUTES] = {
        "positionIn",
        "texcoordIn",
        "normalIn",
        "tangentIn",
        "bitangentIn",
};

Mesh::Mesh(std::string _name, Node * _node) : name_(_name), node_(_node)
{
    show_ = true;
    loadedInGPU_ = false;
}

Mesh::~Mesh()
{
    std::cout << "~Mesh() " << name() << std::endl;
    Graphics::instance().deleteBuffers(name());
}

void Mesh::nodeIs(Node * _node)
{
    node_ = _node;
}

void Mesh::shaderDataIs(ShaderData * _shaderData)
{
    shaderData_ = _shaderData;
}

void Mesh::geometryIs(const std::vector<Vector3> &_position,
                      const std::vector<Vector2> &_texCoord,
                      const std::vector<Vector3> &_normal,
                      const std::vector<Vector3> &_tangent,
                      const std::vector<Vector3> &_bitangent,
                      const std::vector<unsigned int> & _idx)
{
    position_ = _position;
    texCoord_ = _texCoord;
    normal_ = _normal;
    tangent_ = _tangent;
    bitangent_ = _bitangent;
    indices_ = _idx;

    std::vector<Mesh::Vertex> _v;
    generateVertexVector(_v);

    Graphics::instance().buffersNew(name(), VAO_, geometryVBO_, indexVBO_);
    Graphics::instance().geometryIs(geometryVBO_,indexVBO_,_v,_idx,VBO_STATIC);

    const int stride = sizeof(Vertex);
    const int id = shaderData_->shaderID();
    Graphics & g = Graphics::instance();

    int posLoc = g.shaderAttribLoc(id , ShaderAttributes[POSITIONS]);
    int texLoc = g.shaderAttribLoc(id , ShaderAttributes[TEXCOORDS]);
    int normalLoc = g.shaderAttribLoc(id , ShaderAttributes[NORMALS]);
    int tangentLoc = g.shaderAttribLoc(id , ShaderAttributes[TANGENTS]);
    int bitangentLoc = g.shaderAttribLoc(id , ShaderAttributes[BITANGENTS]);

    unsigned int sID = shaderData_->shaderID();

    g.bindGeometry(sID, VAO_, geometryVBO_, 3, stride, posLoc, 0);
    g.bindGeometry(sID, VAO_, geometryVBO_, 2, stride, texLoc, 12);
    g.bindGeometry(sID, VAO_, geometryVBO_, 3, stride, normalLoc, 20);

    //g.bindGeometry(sID, VAO_, geometryVBO_, 3, stride, tangentLoc, 32);
    //g.bindGeometry(sID, VAO_, geometryVBO_, 3, stride, bitangentLoc, 44);

    loadedInGPU_ = true;
}

void Mesh::display() const
{

    if (!show_ || !loadedInGPU_) return;
    unsigned int n = indices_.size();

    const Matrix4 & view = Engine::instance().camera()->viewMtx();
    Matrix4 * modelView = shaderData_->stdMatrix4Data(MODELVIEW);
    *modelView = view * node_->globalModelMtx();

    Matrix4 * model = shaderData_->stdMatrix4Data(MODEL);
    *model = node_->globalModelMtx();

    Matrix3 * normal = shaderData_->stdMatrix3Data(NORMAL);
    *normal = Matrix3(*modelView).inverse().transpose();

    Graphics::instance().drawIndices(VAO_, indexVBO_, n, shaderData_);
}

void Mesh::displayShadowPass(ShaderData * _shaderData) const
{
    if (!show_ || !loadedInGPU_) return;
    unsigned int n = indices_.size();

    const Matrix4 & view = Engine::instance().lightCamera()->viewMtx();
    Matrix4 * modelView = _shaderData->stdMatrix4Data(MODELVIEW);
    *modelView = view * node_->globalModelMtx();


    Graphics::instance().drawIndices(VAO_, indexVBO_, n, _shaderData);
}

void Mesh::generateVertexVector(std::vector<Vertex> & _v)
{
    bool posExists = !position_.empty();
    bool texExists = !texCoord_.empty();
    bool normalExists = !normal_.empty();
    bool tangentExists = !tangent_.empty();
    bool biTangentExists = !bitangent_.empty();

    if (!posExists){
        std::cerr << "No position data in Mesh." << std::endl;
        exit(1);
    }
    _v.resize(position_.size());

    if ((texExists && (position_.size() != texCoord_.size())) ||
        (normalExists && (position_.size() != normal_.size())) ||
        (tangentExists && (position_.size() != tangent_.size())) ||
        (biTangentExists && (position_.size() != bitangent_.size()))) {
        std::cerr << "Position and other Vertex Data missmatch" << std::endl;
        exit(1);
    }

    for (unsigned int i = 0; i < _v.size(); ++i) {
        _v[i].position[0] = position_[i].x;
        _v[i].position[1] = position_[i].y;
        _v[i].position[2] = position_[i].z;
        if (texExists){
            _v[i].texCoords[0] = texCoord_[i].x;
            _v[i].texCoords[1] = texCoord_[i].y;
        }
        if (normalExists){
            _v[i].normal[0] = normal_[i].x;
            _v[i].normal[1] = normal_[i].y;
            _v[i].normal[2] = normal_[i].z;
        }
        if (tangentExists){
            _v[i].tangent[0] = tangent_[i].x;
            _v[i].tangent[1] = tangent_[i].y;
            _v[i].tangent[2] = tangent_[i].z;
        }
        if (biTangentExists){
            _v[i].bitangent[0] = bitangent_[i].x;
            _v[i].bitangent[1] = bitangent_[i].y;
            _v[i].bitangent[2] = bitangent_[i].z;
        }
    }



}

std::vector<Vector3> Mesh::minMax() const {
    float limMax = std::numeric_limits<float>::max();
    float limMin = std::numeric_limits<float>::min();
    Vector3 minCoords = Vector3(limMax, limMax, limMax);
    Vector3 maxCoords = Vector3(limMin, limMin, limMin);
    std::vector<Vector3>::const_iterator it;
    for (it = position_.begin(); it != position_.end(); it++) {
        Vector3 temp = *it;
        if (temp.x > maxCoords.x) maxCoords.x = temp.x;
        else if (temp.x < minCoords.x) minCoords.x = temp.x;
        if (temp.y > maxCoords.y) maxCoords.y = temp.y;
        else if (temp.y < minCoords.y) minCoords.y = temp.y;
        if (temp.z > maxCoords.z) maxCoords.z = temp.z;
        else if (temp.z < minCoords.z) minCoords.z = temp.z;
    }
    std::vector<Vector3> result;
    result.push_back(minCoords);
    result.push_back(maxCoords);
    return result;
}

void Mesh::showIs(bool _show) {
    show_ = _show;
}
