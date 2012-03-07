#include "ParticleSystem.h"

ParticleSystem::ParticleSystem(unsigned int _numEmitters) 
    : numEmitters_(_numEmitters) {}

void ParticleSystem::update(float _dt) {
    std::vector<Emitter*>::iterator it;
    for (it=_emitter.begin(); it!=_emitter.end(); it++) {
        (*it)->update(_dt);
    }
}

void ParticleSystem::newEmitter(unsigned int _numParticles) {
    _emitter.push_back(new Emitter(_numParticles));
}

ParticleSystem::~ParticleSystem() {
    _emitter.clear();
}

