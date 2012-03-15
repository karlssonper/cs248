#include "ParticleSystem.h"
#include "ShaderData.h"
#include "cuda/Emitter.cuh"

ParticleSystem::ParticleSystem(unsigned int _numEmitters) 
    : numEmitters_(_numEmitters) {}

void ParticleSystem::update(float _dt)
{
    std::vector<Emitter*>::iterator it;
    for (it=emitter_.begin(); it!=emitter_.end(); it++) {
        (*it)->update(_dt);
    }
}

Emitter* ParticleSystem::newEmitter(unsigned int _numParticles, ShaderData*_sd)
{
    Emitter* out = new Emitter(_numParticles, _sd);
    emitter_.push_back(out);
    return out;
}

ParticleSystem::~ParticleSystem() {
    _emitter.clear();
}

void ParticleSystem::display() const
{
    for (int i =0; i < emitter_.size(); ++i) {
        emitter_[i]->display();
    }
}
