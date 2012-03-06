/*
 * Mesh.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef MESH_H_
#define MESH_H_

#include <vector>
#include "MathEngine.h"

class Node;
class Mesh
{
    struct Vertex{
        float position[3];
        float texCoords[2];
        float normal[3];
        float tangent[3];
        float bitangent[3];
        unsigned char padding[8];//padding for performance (see sams blog);
    };

public:
    Mesh(std::string _name, Node * _node = NULL);
    const std::string & name() const { return name_;};
    Node * node() const { return node_;};

    void geometryIs(const std::vector<Vector3> &_position,
                    const std::vector<Vector2> &_texCoord,
                    const std::vector<Vector3> &_normal,
                    const std::vector<Vector3> &_tangent,
                    const std::vector<Vector3> &_bitangent);
    void nodeIs(Node * _node);
    void display() const;

private:
    std::string name_;
    Node * node_;

    std::vector<Vector3> position_;
    std::vector<Vector2> texCoord_;
    std::vector<Vector3> normal_;
    std::vector<Vector3> tangent_;
    std::vector<Vector3> bitangent_;

    Mesh();
    Mesh(const Mesh &);
    void operator=(const Mesh &);

    void generateVertexVector(std::vector<Vertex> & _v);
};

#endif /* MESH_H_ */
