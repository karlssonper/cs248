#include "Target.h"
#include "Mesh.h"
#include "HitBox.h"
#include "Node.h"
#include <iostream>

Target::Target(std::string _name, 
               Mesh * _mesh,
               float _energy) 
               : 
               name_(_name),
               mesh_(_mesh),
               energy_(_energy),
               angle_(0.f) {
    std::vector<Vector3> minMax = mesh_->minMax();
    hitBox_ = new HitBox(name_ + "HitBox", minMax.at(0), minMax.at(1));
    updateHitBox();
}

void Target::energyDec(float _e) {
    energy_ -= _e;
}

Target::~Target() {
    delete hitBox_;
}

void Target::updateHitBox() {
    Node * node = mesh_->node();
    Matrix4 localT = node->localModelMtx();
    Matrix4 globalT = node->globalModelMtx();
    Matrix4 transform = globalT*localT;
    hitBox_->p0 = transform*hitBox_->p0;
    hitBox_->p1 = transform*hitBox_->p1;
}