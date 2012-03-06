#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <vector>
#include "Emitter.h"

class ParticleSystem {
public:
    ParticleSystem(unsigned int _numEmitters);
    unsigned int numEmitters() const { return numEmitters_; }
    Emitter* emitter(unsigned int _i) { return _emitter.at(_i); }
    unsigned int first() const { return first_; }
    unsigned int last() const { return last_; }
    void update(float _dt);
    void newEmitter(Emitter::EmitterParams _params);
private:
    ParticleSystem();
    ParticleSystem(const ParticleSystem&);
    unsigned int numEmitters_;
    std::vector<Emitter*> _emitter;
    unsigned int first_;
    unsigned int last_;
};

#endif