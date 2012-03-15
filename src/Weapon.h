#ifndef WEAPON_H
#define WEAPON_H

#include <vector>
#include <string>
#include "Projectile.h"

class Weapon {
public:
    enum WeaponType {
        MACHINEGUN = 0,
        ROCKETLAUNCHER
    };
    Weapon(std::string _name, 
           Vector3 _position,
           float _speed, 
           float _power,
           float _maxDistance);
    std::string name() const { return name_; }
    void fire(Vector3 _direction);
    std::vector<Projectile> projectile;
    void updateProjectiles(float _dt);
    float maxDistance;
    float speed;
    float power;
    Vector3 position;
    WeaponType type;
private:
    std::string name_;
};

#endif