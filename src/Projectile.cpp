#include "Projectile.h"
#include "HitBox.h"

Projectile::Projectile(Vector3 _position, Vector3 _speed, float _maxDist) 
    : position(_position), speed(_speed), maxDist(_maxDist) {}

bool Projectile::hitBoxTest(const HitBox* _hitBox) const {

    Vector3 p0 = _hitBox->p0;
    Vector3 p1 = _hitBox->p1;

    if (p0.x < p1.x) {
        if (position.x < p0.x || position.x > p1.x) return false;
    } else {
        if (position.x < p1.x || position.x > p0.x) return false;
    }

    if (p0.y < p1.y) {
        if (position.y < p0.y || position.y > p1.y) return false;
    } else {
        if (position.y < p1.y || position.y > p0.y) return false;
    }

    if (p0.z < p1.z) {
        if (position.z < p0.z || position.z > p1.z) return false;
    } else {
        if (position.z < p1.z || position.z > p0.z) return false;
    }

    return true;
}