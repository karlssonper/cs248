#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <vector>
#include "Emitter.h"

class ParticleSystem {
public:
    ParticleSystem(unsigned int _numEmitters);
    ~ParticleSystem();
    unsigned int numEmitters() const { return numEmitters_; }
    Emitter* emitter(unsigned int _i) { return _emitter.at(_i); }
    void update(float _dt);
    void newEmitter(Emitter::EmitterParams _params);
private:
    ParticleSystem();
    ParticleSystem(const ParticleSystem&);
    unsigned int numEmitters_;
    std::vector<Emitter*> _emitter;
};

#endif