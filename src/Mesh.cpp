/*
 * Mesh.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Mesh.h"
#include "Node.h"

Mesh::Mesh(std::string _name, Node * _node) : name_(_name), node_(_node)
{

}

void Mesh::nodeIs(Node * _node)
{
    node_ = _node;
}

void Mesh::geometryIs(const std::vector<Vector3> &_position,
                      const std::vector<Vector2> &_texCoord,
                      const std::vector<Vector3> &_normal,
                      const std::vector<Vector3> &_tangent,
                      const std::vector<Vector3> &_bitangent)
{
    position_ = _position;
    texCoord_ = _texCoord;
    normal_ = _normal;
    tangent_ = _tangent;
    bitangent_ = _bitangent;

    std::vector<Mesh::Vertex> _v;
    generateVertexVector(_v);

}

void Mesh::display() const
{

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

    for (int i = 0; i < _v.size(); ++i) {
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
