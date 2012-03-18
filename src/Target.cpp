#include "Target.h"
#include "Engine.h"
#include "Mesh.h"
#include "Camera.h"
#include "HitBox.h"
#include "Node.h"
#include "ParticleSystem.h"
#include "cuda/Emitter.cuh"
#include "Sound.h"
#include <iostream>

    #include <GL/glew.h>
    #include <GL/glut.h>

Target::Target(std::string _name, 
               Mesh * _mesh,
               float _energy) 
               : 
               name_(_name),
               mesh_(_mesh),
               energy_(_energy),
               angle_(0.f),
               active_(false) {
    std::vector<Vector3> minMax = mesh_->minMax();
    hitBox_ = new HitBox(name_ + "HitBox", minMax.at(0), minMax.at(1));
    hitBoxLocal_ = new HitBox(name_+"HitBoxLocal", minMax.at(0), minMax.at(1));
}

void Target::energyDec(float _e) {
    energy_ -= _e;
}

Target::~Target() {
    delete hitBox_;
    delete hitBoxLocal_;
}

void Target::updateHitBox() {
    Node * node = mesh_->node();
    Matrix4 globalT = node->globalModelMtx();
    hitBox_->p0 = globalT*hitBoxLocal_->p0;
    hitBox_->p1 = globalT*hitBoxLocal_->p1;

    //std::cout << std::endl;
    //hitBox_->p0.print();
    //hitBox_->p1.print();

    midPoint_ = Vector3( (hitBox_->p0.x+hitBox_->p1.x)/2.f,
                         (hitBox_->p0.y+hitBox_->p1.y)/2.f,
                         (hitBox_->p0.z+hitBox_->p1.z)/2.f );

   // midPoint_.print();

    for (unsigned int i=0; i<particleSystem_->numEmitters(); ++i) {
        particleSystem_->emitter(i)->posIs(midPoint_);
    }

}

void Target::updatePos(float _dt) {
    mesh_->node()->translate(Vector3(0.f, -heightDiff_+yOffset_, 0.f));
    mesh_->node()->translate(speed_*_dt);

}

void Target::speedIs(Vector3 _speed) {
    speed_ = _speed;
}

void Target::activeIs(bool _active) {
    active_ = _active;
}

void Target::yOffsetIs(float _yOffset) {
    yOffset_ = _yOffset;
}

void Target::heightDiffIs(float _heightDiff) {
    heightDiff_ = _heightDiff;
}

void Target::particleSystemIs(ParticleSystem * _particleSystem) {
    particleSystem_ = _particleSystem;
}

void Target::explode() {
    std::cout << name_ << " EXPLODED" << std::endl;

    Sound::instance().play(Sound::IMPACT, midPoint_);
    Engine::instance().camera()->shake(1.f, 2.f);

    for (unsigned int i=0; i<particleSystem_->numEmitters(); i++) {
        particleSystem_->emitter(i)->burst();
    }

    active_ = false;
    mesh_->showIs(false);
}
