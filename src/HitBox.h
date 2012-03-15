#ifndef HITBOX_H
#define HITBOX_H

#include "MathEngine.h"
#include <string>

class HitBox {
public:
    HitBox(std::string _name, Vector3 _p0, Vector3 _p1);
    std::string name() const { return name_; }
    Vector3 p0, p1;
private:
    std::string name_;
};

#endif