#ifndef PROJECTILE_H
#define PROJECTILE_H

#include "MathEngine.h"

class HitBox;

class Projectile {
public:
    Projectile(Vector3 _position, Vector3 _speed, float _maxDist);
    Vector3 position;
    Vector3 speed;
    float maxDist;
    bool hitBoxTest(const HitBox* _hitBox) const;
};

#endif