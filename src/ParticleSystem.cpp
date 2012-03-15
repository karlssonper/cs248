#include "ParticleSystem.h"
#include "ShaderData.h"
#include "cuda/Emitter.cuh"

ParticleSystem::ParticleSystem(unsigned int _numEmitters) 
    : numEmitters_(_numEmitters) {}

void ParticleSystem::update(float _dt)
{
    std::vector<Emitter*>::iterator it;
    for (it=_emitter.begin(); it!=_emitter.end(); it++) {
        (*it)->update(_dt);
    }
}

Emitter* ParticleSystem::newEmitter(unsigned int _numParticles, ShaderData*_sd)
{
    Emitter* out = new Emitter(_numParticles, _sd);
    _emitter.push_back(out);
    return out;
}

ParticleSystem::~ParticleSystem() {
    _emitter.clear();
}

void ParticleSystem::display() const
{
    for (unsigned int i =0; i < _emitter.size(); ++i) {
        _emitter[i]->display();
    }
}
