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
    void explode();
    void updateHitBox();

    float energy;
 
    HitBox * hitBox() const { return hitBox_; }
    Mesh * mesh() const { return mesh_; }
    std::string name() const { return name_; }
private:
    HitBox * hitBox_;
    Mesh * mesh_;
    std::string name_;
};

#endif