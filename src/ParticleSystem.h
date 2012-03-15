#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <vector>

class Emitter;
class ShaderData;
class ParticleSystem {
public:
    ParticleSystem(unsigned int _numEmitters);
    ~ParticleSystem();
    unsigned int numEmitters() const { return numEmitters_; }
    Emitter* emitter(unsigned int _i) { return _emitter.at(_i); }
    void update(float _dt);
    Emitter* newEmitter(unsigned int _numParticles, ShaderData*_sd);
    void display() const;
private:
    ParticleSystem();
    ParticleSystem(const ParticleSystem&);
    unsigned int numEmitters_;
    std::vector<Emitter*> emitter_;
};

#endif
