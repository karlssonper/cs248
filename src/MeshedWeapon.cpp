#include "MeshedWeapon.h"
#include "MeshedProjectile.h"

#include "Engine.h"
#include "Camera.h"
#include "Sound.h"

MeshedWeapon::MeshedWeapon(Vector3 _position,
                           float _power,
                           float _speed)
                           :
                           position_(_position),
                           power_(_power),
                           speed_(_speed)
                           {}

void MeshedWeapon::fire(Vector3 _direction) {

    // find first non-active projectile
    for (unsigned int i=0; i<projectiles_.size(); ++i) {
        if (!projectiles_.at(i)->active()) {
            Vector3 normDir = _direction.normalize();
            projectiles_.at(i)->activeIs(true);
            projectiles_.at(i)->positionIs(position_);
            projectiles_.at(i)->speedIs(normDir*speed_);
            projectiles_.at(i)->flightDistanceIs(0.f);
            Sound::instance().play(Sound::CANNON, Vector3(0,0,0));
            Engine::instance().camera()->shake(2.f, 3.f);
        }
    }
}

void MeshedWeapon::addProjectile(MeshedProjectile * _meshedProjectile) {
    projectiles_.push_back(_meshedProjectile);
}

MeshedWeapon::~MeshedWeapon() {
    projectiles_.clear();

}

void MeshedWeapon::positionIs(Vector3 _position) {
    position_ = _position;
}
   

void MeshedWeapon::translate(Vector3 _t) {
    position_ = position_ + _t;
}


