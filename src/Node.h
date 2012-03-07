/*
 * Node.h
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#ifndef NODE_H_
#define NODE_H_

#include <list>
#include <string>
#include "MathEngine.h"

class Node
{
public:
    Node(std::string _name, Node * _parent = NULL);
    const std::string & name() const { return name_;};
    const Matrix4 & globalModelMtx() const { return modelTransformGlobal_; };
    const Matrix4 & localModelMtx() const { return modelTransformLocal_; };
    void localModelMtxIs(const Matrix4 &_m);
    void parentIs(Node * _parent);
    void rotateX(float _degrees);
    void rotateY(float _degrees);
    void rotateZ(float _degrees);
    void rotate(float _degrees, const Vector3 &_axis);
    void translate(Vector3 _T);
    void scale(Vector3 _S);
    void update();
private:
    Node();
    Node(const Node &);
    void operator=(const Node &);

    std::string name_;
    Matrix4 modelTransformLocal_;
    Matrix4 modelTransformGlobal_;
    Node * parent_;
    std::list<Node*> children_;
    bool modelTransformLocalChanged_;
    void addChildren(Node* _children);
    void removeChildren(Node * _children);
    void update(bool _needsUpdate);
};
#endif /* NODE_H_ */
