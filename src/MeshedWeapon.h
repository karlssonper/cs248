#ifndef MESHED_WEAPON
#define MESHED_WEAPON

#include <vector>
#include "MathEngine.h"

class MeshedProjectile;

class MeshedWeapon {
public:
    MeshedWeapon(Vector3 _position, 
                 float _power,
                 float _speed);
    ~MeshedWeapon();
    std::vector<MeshedProjectile*> projectiles() { return projectiles_; }
    void fire(Vector3 _direction, float _pitch, float _yaw);
    void addProjectile(MeshedProjectile* _meshedProjectile);
    void positionIs(Vector3 _position);
    void translate(Vector3 _t);
    float power() const { return power_; }
private:
    float power_;
    float speed_;
    Vector3 position_;
    std::vector<MeshedProjectile*> projectiles_;
};

#endif