#ifndef MESHED_PROJECTILE
#define MESHED_PROJECTILE

#include "MathEngine.h"

class Mesh;
class Node;
class HitBox;
class ParticleSystem;
class MeshedProjectile {
public:
    MeshedProjectile(Vector3 _pos,
                     Vector3 _speed, 
                     float _power, 
                     Mesh * _mesh,
                     float _maxDistance,
                     Node * _translationNode,
                     Node * _rotationNode);
    Mesh * mesh() const { return mesh_; }
    Vector3 position() const { return position_; }
    Vector3 speed() const { return speed_; }
    float power() const { return power_; }
    void update(float _dt);
    bool checkCollision(HitBox* _hitBox);
    bool active() { return active_; }
    void activeIs(bool _active);
    void positionIs(Vector3 _position);
    void speedIs(Vector3 _speed);
    void flightDistanceIs(float _flightDistance);
    float flightDistance() const { return flightDistance_; }
    Node * rotationNode() const { return rotationNode_; }
    void resetRotation();
    void pitchIs(float _pitch);
    void yawIs(float _yaw);
    ParticleSystem * particleSystem() const { return particleSystem_; }
    void particleSystemIs(ParticleSystem * _particleSystem);

private:
    bool active_;
    float power_;
    float maxDistance_;
    float flightDistance_;
    Vector3 position_;
    Vector3 speed_;
    Vector3 tailPos_;
    Node * translationNode_;
    Node * rotationNode_;
    Mesh * mesh_;

    ParticleSystem * particleSystem_;

    float pitch_;
    float yaw_;

};

#endif
