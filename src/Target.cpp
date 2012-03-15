#include "Target.h"
#include <iostream>

Target::Target(std::string _name, 
               Vector3 _position, 
               Vector3 _hbp0, 
               Vector3 _hbp1, 
               float _energy) 
               : 
               name_(_name),
               hitBox(_name + "HitBox", _hbp0, _hbp1),
               position(_position), 
               energy(_energy) {
}

void Target::explode() {
}