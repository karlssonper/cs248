#include "Weapon.h"

Weapon::Weapon(std::string _name, 
               Vector3 _position, 
               float _speed, 
               float _power,
               float _maxDistance) 
               : 
               name_(_name), 
               position(_position), 
               speed(_speed), 
               power(_power),
               maxDistance(_maxDistance) 
               {}

void Weapon::fire(Vector3 _direction)  {

    Vector3 speedVector = _direction;
    speedVector.normalize();
    speedVector = speedVector*speed;
   
    projectile.push_back(Projectile(position, speedVector, maxDistance));

}

void Weapon::updateProjectiles(float _dt) {
    std::vector<Projectile>::iterator it = projectile.begin();
    while (it!=projectile.end()) {
        float dx = it->speed.x * _dt;
        float dy = it->speed.y * _dt;
        float dz = it->speed.z * _dt;
        it->position.x += dx;
        it->position.y += dy;
        it->position.z += dz;
        it->maxDist -= sqrt(dx*dx + dy*dy + dz*dz);
        if (it->maxDist < 0.f) { 
            it = projectile.erase(it);
        } else {
            ++it;
        }
    }
}
