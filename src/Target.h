#ifndef TARGET_H
#define TARGET_H

#include "MathEngine.h"
#include <string>

class Mesh;
class HitBox;
class Target {
public:
    Target(std::string _name, 
           Mesh * _mesh,
           float _energy);
    ~Target();
    void updateHitBox();
    float energy() const { return energy_; }
    void energyDec(float _e);
    float angle() const { return angle_; }
    Vector3 frontAnchor() const { return frontAnchor_; }
    Vector3 backAnchor() const { return backAnchor_; }
    Vector3 middleAnchor() const { return middleAnchor_; }
    HitBox * hitBox() const { return hitBox_; }
    Mesh * mesh() const { return mesh_; }
    std::string name() const { return name_; }
private:
    float energy_;
    Vector3 frontAnchor_;
    Vector3 backAnchor_;
    Vector3 middleAnchor_;
    float angle_;
    HitBox * hitBox_;
    Mesh * mesh_;
    std::string name_;
};

#endif