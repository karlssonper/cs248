#ifndef TARGET_H
#define TARGET_H

#include "HitBox.h"
#include "MathEngine.h"
#include <string>

class Target {
public:
    Target(std::string _name, 
           Vector3 _position,
           Vector3 _hbp0, 
           Vector3 _hbp1, 
           float _energy);
   
    void explode();
  
    Vector3 position;
    float energy;
    HitBox hitBox;
    std::string name() const { return name_; }
private:
    std::string name_;
};

#endif