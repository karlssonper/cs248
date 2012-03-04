/*
 * Node.cpp
 *
 *  Created on: Mar 2, 2012
 *      Author: per
 */

#include "Node.h"

Node::Node(std::string _name, Node * _parent) : name_(_name)
{
    children_.resize(0);
    parentIs(_parent);
    modelTransformLocalChanged_ = false;
}

void Node::parentIs(Node * _parent)
{
    if (parent_ != NULL) {
        parent_->removeChildren(this);
    }
    parent_ = _parent;
    parent_->addChildren(this);
}

void Node::addChildren(Node * _children)
{
    children_->push_back(_children);
}

void Node::removeChildren(Node * _children)
{
    children_->remove(_children);
}

void Node::rotateX(float _degrees)
{
    modelTransformLocal_ =
            modelTransformLocal_ * Matrix4::rotate(_degrees, 1.0f, 0.0f, 0.0f);
    modelTransformLocalChanged_ = true;
}

void Node::rotateY(float _degrees)
{
    modelTransformLocal_=
            modelTransformLocal_ * Matrix4::rotate(_degrees, 0.0f, 1.0f, 0.0f);
    modelTransformLocalChanged_ = true;
}

void Node::rotateZ(float _degrees)
{
    modelTransformLocal_ =
            modelTransformLocal_ * Matrix4::rotate(_degrees, 0.0f, 0.0f, 1.0f);
    modelTransformLocalChanged_ = true;
}

void Node::rotate(float _degrees, const Vector3 &_axis)
{
    modelTransformLocal_ = modelTransformLocal_ * Matrix4::rotate(
                                                        _degrees,
                                                        _axis.x,
                                                        _axis.y,
                                                        _axis.z
                                                        );
    modelTransformLocalChanged_ = true;
}

void Node::translate(Vector3 _T)
{
    modelTransformLocal_ =
            modelTransformLocal_ * Matrix4::translate( _T.x, _T.y, _T.z);
    modelTransformLocalChanged_ = true;
}
void Node::scale(vector3 _S)
{
    modelTransformLocal_ =
            modelTransformLocal_ * Matrix4::scale( _S.x, _S.y, _S.z);
    modelTransformLocalChanged_ = true;
}

void Node::update()
{
    for (std::list<Node*>::iterator it = children_.begin(); it != children_.end(); ++it){
        (*it)->update(false);
    }
}

void Node::update(bool _needsUpdate)
{
    const bool upd = modelTransformLocalChanged_ || _needsUpdate;

    if (upd) {
        modelTransformGlobal_ = parent_->modelMtx() * modelTransformLocal_;
    }
    for (std::list<Node*>::iterator it = children_.begin(); it != children_.end(); ++it){
        (*it)->update(upd);
    }
    modelTransformLocalChanged_ = false;
}
