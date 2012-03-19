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

    Vector3 p0 = hitBox_->p0;
    Vector3 p1 = hitBox_->p1;

    midPoint_ = Vector3( (p0.x+p1.x)/2.f,
                         (p0.y+p1.y)/2.f,
                         (p0.z+p1.z)/2.f );

    // calculate the two points to put wakes at the front
    // somewhere betweeen the midpoint and the two corner points
    // (this is hardcoded for now)

    // smallest Z is the head direction
    float frontZ;
    if (p0.z < p1.z) frontZ = p0.z;
    else frontZ = p1.z;
    frontLeft_.z = frontRight_.z = (2.f*frontZ + 3.f*midPoint_.z) / 5.f;

    // smallest X is 'left'
    float leftX, rightX;
    if (p0.x < p1.x) { leftX = p0.x; rightX = p1.x; }
    else { leftX = p1.x; rightX = p0.x; }
    frontLeft_.x = (2.f*leftX + 3.f*midPoint_.x) / 5.f;
    frontRight_.x = (2.f*rightX + 3.f*midPoint_.x) / 5.f;

    // smallest Y is lowest
    float frontY;
    if (p0.y < p1.y) frontY = p0.y;
    else frontY = p1.y;
    frontLeft_.y = frontRight_.y = frontY + 2.f;

    for (unsigned int i=0; i<explosionPs_->numEmitters(); ++i) {
        explosionPs_->emitter(i)->posIs(midPoint_);
    }

    foamPs_->emitter(0)->posIs(frontLeft_);
    foamPs_->emitter(1)->posIs(frontRight_);
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

void Target::explosionPsIs(ParticleSystem * _explosionPs) {
    explosionPs_ = _explosionPs;
}

void Target::foamPsIs(ParticleSystem * _foamPs) {
    foamPs_ = _foamPs;
}

void Target::explode() {

    Sound::instance().play(Sound::IMPACT, midPoint_);
    Engine::instance().camera()->shake(1.f, 2.f);

    for (unsigned int i=0; i<explosionPs_->numEmitters(); i++) {
        explosionPs_->emitter(i)->burst();
    }

    active_ = false;
    mesh_->showIs(false);
}
