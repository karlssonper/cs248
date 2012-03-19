#ifndef TARGET_H
#define TARGET_H

#include "MathEngine.h"
#include <string>
#include <vector>

class Mesh;
class HitBox;
class ParticleSystem;
class Emitter;
class Target {
public:
    Target(std::string _name, 
           Mesh * _mesh,
           float _energy);
    ~Target();
    float energy() const { return energy_; }
    void energyDec(float _e);
    void updatePos(float _dt);
    void updateHitBox();
    float angle() const { return angle_; }
    HitBox * hitBox() const { return hitBox_; }
    Mesh * mesh() const { return mesh_; }
    std::string name() const { return name_; }
    Vector3 speed() const { return speed_; }
    bool active() const { return active_; }
    void activeIs(bool _active);
    void speedIs(Vector3 _speed);
    Vector3 midPoint() const { return midPoint_; }
    Vector3 frontLeft() const { return frontLeft_; }
    Vector3 frontRight() const { return frontRight_; }
    void explode();

    ParticleSystem * particleSystem() const { return particleSystem_; }
    void particleSystemIs(ParticleSystem * _particleSystem);
    

    float heightDiff() const { return heightDiff_; }
    float yOffset() const { return yOffset_; }
    void heightDiffIs(float _heightDiff);
    void yOffsetIs(float _yOffset);


private:

    float energy_;
    Vector3 midPoint_;
    Vector3 frontLeft_;
    Vector3 frontRight_;
    Vector3 speed_;
    float angle_;
    HitBox * hitBoxLocal_;
    HitBox * hitBox_;
    Mesh * mesh_;
    std::string name_;
    bool active_;

    ParticleSystem * particleSystem_;

    float heightDiff_;
    float yOffset_;

};

#endif
