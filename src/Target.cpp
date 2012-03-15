#include "Target.h"
#include "Mesh.h"
#include "HitBox.h"
#include <iostream>

Target::Target(std::string _name, 
               Mesh * _mesh,
               float _energy) 
               : 
               name_(_name),
               mesh_(_mesh),
               energy(_energy) {
    std::vector<Vector3> minMax = mesh_->minMax();
    hitBox_ = new HitBox(name_ + "HitBox", minMax.at(0), minMax.at(1));
    updateHitBox();
}

void Target::explode() {
}

Target::~Target() {
    delete hitBox_;
}

void Target::updateHitBox() {
}