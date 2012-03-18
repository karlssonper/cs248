#include "MeshedProjectile.h"
#include "Mesh.h"
#include "HitBox.h"
#include "Node.h"

MeshedProjectile::MeshedProjectile(Vector3 _pos,
                                   Vector3 _speed,
                                   float _power,
                                   Mesh * _mesh,
                                   float _maxDistance)
                                   :
                                   position_(_pos),
                                   power_(_power),
                                   mesh_(_mesh),
                                   maxDistance_(_maxDistance),
                                   flightDistance_(0.f),
                                   active_(false) 
                                   {}

void MeshedProjectile::update(float _dt) {
    if (!active_) return;
    Vector3 d = speed_*_dt;

    mesh_->node()->translate(d);

    Node * node = mesh_->node();
    Matrix4 globalT = node->globalModelMtx();
    position_ = globalT*position_;

    flightDistance_ += d.mag();
    std::cout << "flightDist :" << flightDistance_ << std::endl;

    if (flightDistance_ > maxDistance_) active_ = false;
}

bool MeshedProjectile::checkCollision(HitBox* _hitBox) {

    Vector3 p0 = _hitBox->p0;
    Vector3 p1 = _hitBox->p1;

    if (p0.x < p1.x) {
        if (position_.x < p0.x || position_.x > p1.x) return false;
    } else {
        if (position_.x < p1.x || position_.x > p0.x) return false;
    }

    if (p0.y < p1.y) {
        if (position_.y < p0.y || position_.y > p1.y) return false;
    } else {
        if (position_.y < p1.y || position_.y > p0.y) return false;
    }

    if (p0.z < p1.z) {
        if (position_.z < p0.z || position_.z > p1.z) return false;
    } else {
        if (position_.z < p1.z || position_.z > p0.z) return false;
    }

    return true;
}

void MeshedProjectile::activeIs(bool _active) {
    active_ = _active;
}

void MeshedProjectile::positionIs(Vector3 _position) {

    Vector3 diff = position_ -  _position;
    mesh_->node()->translate(diff);

    Node * node = mesh_->node();
    Matrix4 globalT = node->globalModelMtx();
    position_ = globalT*position_;

}

void MeshedProjectile::speedIs(Vector3 _speed) {
    speed_ = _speed;
}

void MeshedProjectile::flightDistanceIs(float _flightDistance) {
    flightDistance_ = _flightDistance;
}